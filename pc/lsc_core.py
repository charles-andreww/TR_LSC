"""
la definició ÚNICA de les dades dels guants.

Tots els programes que treballen amb les dades importen aquest fitxer:
    lsc_app.py        (calibratge, enregistrament guiat i prova)
    build_dataset.py  (converteix les gravacions en finestres d'entrenament per a Edge Impulse)
    proves            (comproven que el codi en C del guant dret dona els mateixos números)

El guant dret calcula les dades d'entrada del model amb firmware/LSC_Right/lsc_features.h.
Aquell fitxer en C és una còpia, línia per línia, de la funció features() d'aquí sota, i les
proves comproven que totes dues donen el mateix resultat.

QUÈ VEU EL MODEL (34 números per trama, 50 trames per segon, finestra de 100 trames = 2,0 s)

Per a cada mà (primer la dreta i després l'esquerra), 17 números:

  polze índex cor anular petit    flexió de cada dit: 0 = la mà oberta del calibratge, 1 = el puny tancat
                                  (limitada entre -0,2 i 1,2)
  xx xy xz                        direcció de l'eix X del guant (cap on apunten els dits)  } orientació de la mà
  zx zy zz                        direcció de l'eix Z del guant (perpendicular al palmell) } respecte de la
                                  gravetat i de la direcció "endavant" de la persona.
                                  Són valors continus: no fan salts de +180 a -180 graus i no tenen
                                  ambigüitats (un quaternió q i -q són la mateixa orientació).
                                  "Endavant" = cap on apuntaven les mans en la posició de zero
                                  (mans planes, dits cap a la pantalla).
  wx wy wz                        velocitat de gir de la mà (giroscopi, rad/s dividits per 8)
  ax ay az                        acceleració de la mà sense la gravetat, en el sistema endavant/esquerra/amunt
                                  (en g dividits per 2)

Per què aquests valors i no els angles (balanceig, capcineig, guinyada) i el quaternió de la primera versió:
  * els angles salten de +180 a -180 graus i es trenquen quan la mà apunta cap amunt: el model veu salts
    enormes que no són reals;
  * un quaternió i el seu negatiu són la mateixa orientació, però els números són completament diferents;
  * el zero antic era l'orientació COMPLETA de la mà en prémer Zero, de manera que una posició de zero una
    mica diferent canviava tots els valors d'orientació. Ara de la posició de zero només se n'agafa la
    direcció "endavant"; la direcció amunt/avall surt de la gravetat, que és la mateixa cada dia.
"""
import json
import math
import os
import time

import numpy as np

FS = 50                 # trames per segon
FRAME_MS = 20           # durada d'una trama (ms)
WINDOW = 100            # trames d'una finestra del model (2,0 s)
STRIDE = 10             # trames entre dues execucions del model (200 ms)
GYRO_SCALE = 8.0        # el gir (rad/s) es divideix per 8 per passar-lo a unitats del model
ACC_SCALE = 2.0         # l'acceleració (g) es divideix per 2
FLEX_MIN, FLEX_MAX = -0.2, 1.2      # límits de la flexió normalitzada
MIN_FLEX_SPAN = 20.0    # diferència mínima (comptes de l'ADC) entre mà oberta i puny: per sota, el sensor no serveix
FW_VERSION = "2.0"      # versió del programa dels guants que esperem

# Noms de les columnes. Dades en cru d'una mà: 5 dits, quaternió, gir, acceleració i indicadors.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RAW_HAND = ["f0", "f1", "f2", "f3", "f4", "qw", "qx", "qy", "qz",
            "gx", "gy", "gz", "ax", "ay", "az", "flags"]
#temps, número de trama, mà dreta (r_), mà esquerra (l_) i antiguitat de les dades de l'esquerra.
RAW_COLUMNS = ["t_ms", "seq"] + ["r_" + c for c in RAW_HAND] + ["l_" + c for c in RAW_HAND] + ["l_age"]
# Les 17 dades d'entrada del model per a cada mà (vegeu l'explicació del principi).
FEAT_HAND = FINGERS + ["xx", "xy", "xz", "zx", "zy", "zz", "wx", "wy", "wz", "ax", "ay", "az"]
FEAT_COLUMNS = ["r_" + c for c in FEAT_HAND] + ["l_" + c for c in FEAT_HAND]
N_FEAT = len(FEAT_COLUMNS)          # 34
FLAG_IMU, FLAG_STILL, FLAG_LEFT_OK = 1, 2, 4     # la IMU funciona / el guant està quiet / l'esquerre arriba bé
LEFT_MAX_AGE_MS = 250               # dades del guant esquerre més antigues que això = guant esquerre perdut


# Protocol de comunicació
# Cada línia de dades acaba amb *HH = XOR de tots els caràcters anteriors a '*', en 2 xifres hexadecimals.
# Guant -> PC   L,núm,<16 valors>*HH                       guant esquerre, 50 Hz (l'ordinador el reenvia al dret)
#               F,núm,<16 dreta>,<16 esquerra>,l_age*HH    guant dret, 50 Hz: exactament el que fa servir el model
#               P,núm,<p0..pN-1>,signe*HH                  guant dret, cada 200 ms: sortida del model i decisió
#               {"type":...}                               línies d'estat en format JSON
# PC -> dret    C,<dreta: obert5,tancat5,psi0>,<esquerra: obert5,tancat5,psi0>*HH   calibratge
#               ?                                          "qui ets?" (respon {"type":"hello",...})

#combina tots els caràcters amb l'operació XOR i la torna en hexadecimal.
def checksum(body: str) -> str:
    x = 0
    for ch in body.encode("ascii", "replace"):
        x ^= ch
    return "%02X" % x


# "Segella" una línia: hi afegeix "*HH".
def seal(body: str) -> str:
    return body + "*" + checksum(body)


def unseal(line: str):
    """Torna el contingut d'una línia si la suma de verificació és correcta, o None si falta o no coincideix."""
    line = line.strip()
    star = line.rfind("*")
    if star < 0 or len(line) - star != 3:
        return None
    body = line[:star]
    return body if checksum(body) == line[star + 1:].upper() else None


def parse_frame(line: str):
    """Línia F -> llista de números en l'ordre de RAW_COLUMNS, sense t_ms (primer el número de trama), o None."""
    body = unseal(line)
    if body is None or not body.startswith("F,"):
        return None
    parts = body.split(",")
    if len(parts) != 1 + 1 + 16 + 16 + 1:      # "F", número, 16 valors de cada mà i antiguitat
        return None
    try:
        return [float(v) for v in parts[1:]]
    except ValueError:
        return None


def parse_probs(line: str):
    """Línia P -> (número de trama, [probabilitats], posició del signe que s'ha de dir o -1), o None."""
    body = unseal(line)
    if body is None or not body.startswith("P,"):
        return None
    parts = body.split(",")
    try:
        return int(parts[1]), [float(v) for v in parts[2:-1]], int(parts[-1])
    except (ValueError, IndexError):
        return None


# Prepara la línia C que s'envia al guant dret amb el calibratge de les dues mans.
def calibration_command(cal: dict) -> str:
    vals = []
    for hand in ("right", "left"):
        h = cal[hand]
        vals += ["%.2f" % v for v in h["open"]] + ["%.2f" % v for v in h["closed"]] + ["%.6f" % h["psi0"]]
    return seal("C," + ",".join(vals))


# Càlculs d'orientació
def quat_to_matrix(q):
    """Converteix quaternions (w, x, y, z; del guant cap al món, amb Z del món amunt, com al filtre de
    Mahony) en matrius de rotació de 3x3. Cada columna de la matriu és un eix del guant vist des del món."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(n > 1e-9, n, 1.0)                         # longitud 1 (sense dividir per zero)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_mean(qs):
    """Orientació mitjana de molts quaternions. Abans de fer la mitjana es giren els que tenen el signe
    contrari, perquè q i -q són la mateixa orientació."""
    qs = np.asarray(qs, dtype=np.float64)
    ref = qs[len(qs) // 2]                                     # un quaternió de referència (el del mig)
    s = np.where((qs @ ref) < 0, -1.0, 1.0)[:, None]
    m = (qs * s).mean(axis=0)
    return m / np.linalg.norm(m)


def heading_of(q):
    """Direcció "endavant" (en radians, al voltant de l'eix vertical) d'una mà més o menys plana.
    Fa servir l'eix X del guant (o l'eix Y, si l'X és gairebé vertical) projectat sobre el pla horitzontal."""
    R = quat_to_matrix(q)
    hx, hy = R[:2, 0], R[:2, 1]                                # eixos X i Y del guant, només la part horitzontal
    if np.hypot(*hx) >= np.hypot(*hy):
        return float(math.atan2(hx[1], hx[0]))
    return float(math.atan2(hy[1], hy[0]) - math.pi / 2)


# Angle (en graus) entre dues orientacions.
def angle_between(q1, q2):
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


# Dades d'entrada (= lsc_features.h)
def hand_features(flex, q, gyro, acc, cal_hand):
    """Calcula els 17 valors d'una mà per a moltes trames alhora.
    flex (N,5) en comptes de l'ADC, q (N,4), gyro (N,3) en rad/s, acc (N,3) en g  ->  (N,17) en unitats del model."""
    flex = np.asarray(flex, dtype=np.float64)
    op = np.asarray(cal_hand["open"], dtype=np.float64)
    cl = np.asarray(cal_hand["closed"], dtype=np.float64)
    # Flexió dels dits: com més doblegat, més baixa és la lectura; per això es fa (obert - lectura).
    bend = np.clip((op - flex) / (op - cl), FLEX_MIN, FLEX_MAX)

    # Gir al voltant de l'eix vertical perquè "endavant" sigui la direcció de la posició de zero.
    c, s = math.cos(cal_hand["psi0"]), math.sin(cal_hand["psi0"])
    Rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])       # gira el món perquè "endavant" sigui +X
    Rb = Rz @ quat_to_matrix(q)                                        # del guant al sistema del cos
    xcol, zcol = Rb[..., :, 0], Rb[..., :, 2]                          # eix X i eix Z del guant
    w = np.asarray(gyro, dtype=np.float64) / GYRO_SCALE               # velocitat de gir
    a = np.einsum("nij,nj->ni", Rb, np.asarray(acc, dtype=np.float64)) / ACC_SCALE   # acceleració en el sistema del cos
    return np.concatenate([bend, xcol, zcol, w, a], axis=1)


def features(raw, cal):
    """raw: files gravades (N x columnes de RAW_COLUMNS); cal: calibratge -> (N,34): primer la mà dreta i després l'esquerra."""
    raw = np.asarray(raw, dtype=np.float64)
    col = {n: i for i, n in enumerate(RAW_COLUMNS)}
    out = []
    for p, hand in (("r_", "right"), ("l_", "left")):
        g = lambda names: raw[:, [col[p + n] for n in names]]           # agafa unes columnes d'aquesta mà
        out.append(hand_features(g(["f0", "f1", "f2", "f3", "f4"]), g(["qw", "qx", "qy", "qz"]),
                                 g(["gx", "gy", "gz"]), g(["ax", "ay", "az"]), cal[hand]))
    return np.concatenate(out, axis=1)


def frames_valid(raw):
    """True a les trames en què les dues IMU funcionen i les dades del guant esquerre són recents
    (la mateixa regla que fa servir el guant dret)."""
    raw = np.asarray(raw)
    col = {n: i for i, n in enumerate(RAW_COLUMNS)}
    rf = raw[:, col["r_flags"]].astype(int)
    lf = raw[:, col["l_flags"]].astype(int)
    return ((rf & FLAG_IMU) > 0) & ((lf & FLAG_IMU) > 0) & ((rf & FLAG_LEFT_OK) > 0) & \
           (raw[:, col["l_age"]] <= LEFT_MAX_AGE_MS)


# Calibratge
def zero_heading(raw_rows):
    """Posició de zero (mans planes, dits endavant, quietes) -> per a cada mà, psi0 (direcció "endavant")
    i dues mesures de qualitat: quant s'ha mogut la mà (wander_deg) i quant estava inclinada (tilt_deg)."""
    raw = np.asarray(raw_rows, dtype=np.float64)
    col = {n: i for i, n in enumerate(RAW_COLUMNS)}
    res = {}
    for p, hand in (("r_", "right"), ("l_", "left")):
        qs = raw[:, [col[p + n] for n in ("qw", "qx", "qy", "qz")]]
        qm = quat_mean(qs)                                         # orientació mitjana durant la captura
        wander = max(angle_between(qm, q) for q in qs)             # la trama que més s'allunya de la mitjana
        R = quat_to_matrix(qm)
        tilt_x = math.degrees(math.asin(min(1.0, abs(R[2, 0]))))  # inclinació de l'eix X respecte de l'horitzontal
        tilt_y = math.degrees(math.asin(min(1.0, abs(R[2, 1]))))  # inclinació de l'eix Y
        res[hand] = {"psi0": heading_of(qm), "wander_deg": wander, "tilt_deg": min(tilt_x, tilt_y)}
    return res


def flex_calibration(open_reps, fist_reps, min_good=3, tol=0.25):
    """Calibratge dels dits d'UNA mà. open_reps / fist_reps: una llista (una entrada per repetició) de
    lectures de l'ADC (n x 5) amb la mà oberta i amb el puny tancat.
    Per a cada dit es fa la mitjana de les repeticions que coincideixen; una mà poc oberta o un puny
    poc tancat es descarten. Torna (obert[5], tancat[5], informació) o dona un error que diu quin dit falla."""
    op, cl, info = [], [], []
    for f in range(5):
        o = np.array([np.mean(r[:, f]) for r in open_reps])      # lectura mitjana de cada repetició (mà oberta)
        c = np.array([np.mean(r[:, f]) for r in fist_reps])      # i amb el puny tancat
        mo = np.array([np.ptp(r[:, f]) for r in open_reps])      # quant ha variat dins de cada repetició
        mc = np.array([np.ptp(r[:, f]) for r in fist_reps])
        d = 1.0 if np.median(o) >= np.median(c) else -1.0      # en aquest maquinari, la mà oberta llegeix més alt
        so, sc = np.sort(d * o)[::-1], np.sort(d * c)          # de la més oberta a la menys / del puny més tancat al menys
        ao = d * so[min(1, len(so) - 1)]                       # es pren la 2a més oberta i el 2n més tancat:
        ac = d * sc[min(1, len(sc) - 1)]                       # així una sola repetició estranya no fa de referència
        span = abs(ao - ac)                                    # diferència entre mà oberta i puny
        if span < MIN_FLEX_SPAN:
            raise ValueError("range:%s:%.0f" % (FINGERS[f], span))       # el sensor gairebé no canvia
        still = max(0.5 * span, 40.0)                          # variació màxima admesa dins d'una repetició
        # Repeticions bones: prop de la referència i sense moure's gaire.
        go = [k for k in range(len(o)) if abs(o[k] - ao) <= tol * span and mo[k] <= still]
        gc = [k for k in range(len(c)) if abs(c[k] - ac) <= tol * span and mc[k] <= still]
        if len(go) < min_good or len(gc) < min_good:
            raise ValueError("inconsistent:%s:%d/%d" % (FINGERS[f], len(go), len(gc)))   # massa poques repeticions bones
        op.append(float(np.mean(o[go])))
        cl.append(float(np.mean(c[gc])))
        info.append({"finger": FINGERS[f], "span": round(abs(op[-1] - cl[-1]), 1),
                     "good_open": len(go), "good_fist": len(gc)})
    return op, cl, info


# On és el signe dins d'una presa?
def motion_energy(feat, hand="r"):
    """Quant es mou una mà a cada trama: velocitat de gir (rad/s) + 0,3 x velocitat de flexió dels dits (1/s)
    + 3 x acceleració (g)."""
    off = 0 if hand == "r" else 17                             # columnes de la mà dreta (0-16) o de l'esquerra (17-33)
    gyro = np.linalg.norm(feat[:, off + 11: off + 14] * GYRO_SCALE, axis=1)
    acc = np.linalg.norm(feat[:, off + 14: off + 17] * ACC_SCALE, axis=1)
    bend = feat[:, off: off + 5]
    k = np.ones(5) / 5                                         # mitjana de 5 trames per suavitzar
    sm = np.stack([np.convolve(bend[:, i], k, mode="same") for i in range(5)], axis=1)
    vel = np.zeros_like(sm)
    vel[5:] = (sm[5:] - sm[:-5]) / (5.0 / FS)                  # velocitat de flexió: canvi en 5 trames (0,1 s)
    e = gyro + 0.3 * np.abs(vel).sum(axis=1) + 3.0 * acc
    return np.convolve(e, k, mode="same")


def pose_distance(feat, ref, hand="r"):
    """Com d'allunyades estan la forma i l'orientació de la mà d'una trama de referència (per exemple, el repòs).
    Fa servir els 5 dits i els dos eixos del guant (11 valors)."""
    off = 0 if hand == "r" else 17
    cols = list(range(off, off + 11))
    return np.linalg.norm(feat[:, cols] - ref[cols], axis=1)


def close_gaps(mask, n):
    """Omple els buits (False) de com a màxim n trames que hi ha entre dos True."""
    m = np.array(mask, dtype=bool)
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= n + 1:
            m[a:b] = True
    return m


def _runs(mask):
    """Llista de (inici, final) de cada tram seguit de True (el final no s'inclou)."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def analyse_take(feat, cue, sign):
    """Busca el signe dins d'una presa gravada.

    feat: dades d'entrada (N x 34), cue: trama del senyal "ARA!", sign: entrada de signs.json.
    Torna un diccionari amb ok / reason (el motiu en català, que es mostra just després de la presa) i:
      start, end         tot el moviment, des que la mà surt del repòs fins que hi torna
      hold_start/_end    només per a les lletres: la part en què la lletra es manté quieta
    """
    N = len(feat)
    two = int(sign.get("hands", 1)) == 2                       # signe de dues mans?
    b0, b1 = max(0, cue - FS), max(1, cue - 5)                 # el segon abans del senyal = posició de repòs
    res = {"ok": False, "reason": "", "start": None, "end": None, "hold_start": None, "hold_end": None}
    if b1 - b0 < 10 or N <= cue:
        res["reason"] = "falta el temps de repòs abans del senyal"
        return res
    rest = feat[b0:b1].mean(axis=0)                            # com és el repòs d'aquesta presa
    hands = ("r", "l") if two else ("r",)                      # en els signes d'una mà només es mira la dreta
    e = np.max([motion_energy(feat, h) for h in hands], axis=0)        # quant es mou
    d = np.max([pose_distance(feat, rest, h) for h in hands], axis=0)  # quant s'allunya del repòs
    res["left_peak"] = float(motion_energy(feat, "l")[cue:].max())
    if e[b0:b1].mean() > 0.8:
        res["reason"] = "no estaves quiet abans del senyal"
        return res
    # Llindars a partir del repòs d'aquesta presa: el que es mou o s'allunya clarament més que el repòs és "actiu".
    thr = max(0.8, float(e[b0:b1].mean() + 5 * e[b0:b1].std()))
    dthr = max(0.35, float(d[b0:b1].mean() + 5 * d[b0:b1].std()))
    res["threshold"] = thr
    active = (e > thr) | (d > dthr)
    runs = [r for r in _runs(active[b1:]) if r[1] - r[0] >= 3]   # trams actius d'almenys 3 trames
    if not runs:
        res["reason"] = "no s'ha detectat cap moviment"
        return res
    runs = [(a + b1, b + b1) for a, b in runs]
    start, end = runs[0]
    for a, b in runs[1:]:                                       # un signe es pot aturar un moment pel mig:
        if a - end < int(0.4 * FS):                             # pauses de menys de 0,4 s s'ajunten
            end = b
        else:
            break
    res["start"], res["end"] = int(start), int(end)
    dur = (end - start) / FS                                    # durada del moviment (s)
    # Comprovacions: començar a temps, tornar a repòs, durada raonable.
    if start < cue - int(0.3 * FS):
        res["reason"] = "has començat abans del senyal"
        return res
    if start > cue + int(1.5 * FS):
        res["reason"] = "has començat massa tard"
        return res
    if N - end < int(0.3 * FS):
        res["reason"] = "no has tornat a repòs a temps (o has deixat les mans en una altra posició)"
        return res
    if dur < 0.3:
        res["reason"] = "moviment massa curt"
        return res

    if sign.get("type") == "static":
        # Lletres: la part quieta més llarga dins del moviment, amb la mà clarament fora del repòs.
        still = (e < thr) & (d > dthr)                         # tan quieta com en repòs, però sense ser-hi
        still = close_gaps(still, 5)                            # una sacsejada de 0,1 s no trenca la part quieta
        holds = [(a, b) for a, b in _runs(still[start:end]) if b - a > 0]
        if not holds:
            res["reason"] = "no s'ha vist la lletra quieta"
            return res
        a, b = max(holds, key=lambda r: r[1] - r[0])
        res["hold_start"], res["hold_end"] = int(start + a), int(start + b)
        if (b - a) / FS < 0.9:
            res["reason"] = "lletra mantinguda massa poc (%.1f s, cal 1 s)" % ((b - a) / FS)
            return res
    else:
        # Signes amb moviment, de repòs a repòs ha de durar menys de 2,5 s (si no, no cap en una finestra de 2 s).
        if dur > 2.5:
            res["reason"] = "massa lent (%.1f s): de repòs a repòs ha de durar menys de 2.5 s" % dur
            return res
    res["ok"] = True
    res["reason"] = "ok (%.1f s)" % dur
    return res


# Fitxers
# Llegeix signs.json (el vocabulari). Torna el fitxer sencer i un diccionari clau -> signe.
def load_signs(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "signs.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    signs = {s["key"]: s for s in data["signs"]}
    return data, signs


def save_take(folder, label, session, raw_rows, meta):
    """Desa una gravació: un CSV amb les trames en cru i un JSON amb tot el que cal per tornar a calcular
    les dades d'entrada (calibratge, moments dels senyals, preses rebutjades...)."""
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = "%s__%s__%s_%03d" % (label, session, stamp, int(time.time() * 1000) % 1000)   # signe__sessió__data
    with open(os.path.join(folder, base + ".csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(RAW_COLUMNS) + "\n")
        for r in raw_rows:
            fh.write(",".join(_fmt(v) for v in r) + "\n")
    meta = dict(meta, label=label, session=session, file=base + ".csv", fw=FW_VERSION,
                frames=len(raw_rows), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(folder, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return base


# Escriu un número sense decimals sobrers (els enters, sense decimals).
def _fmt(v):
    if float(v).is_integer():
        return str(int(v))
    return ("%.5f" % v).rstrip("0").rstrip(".")


# Llegeix una gravació, torna el JSON i les trames del CSV.
def load_take(json_path):
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    raw = np.loadtxt(os.path.join(os.path.dirname(json_path), meta["file"]), delimiter=",", skiprows=1, ndmin=2)
    return meta, raw
