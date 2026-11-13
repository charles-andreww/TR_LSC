
"""
converteix les gravacions en dades d'entrenament per a Edge Impulse.

    python build_dataset.py                      # dataset/raw  ->  dataset/edge_impulse/{training,testing}
    python build_dataset.py --check              # a més, entrena aquí un model ràpid i mostra què es confon
    python build_dataset.py --test-session 20261105-1730_enzo   # reserva una sessió sencera per al test

Cada bloc gravat conté diverses preses d'un signe, amb la trama exacta de cada senyal "ARA!".
Per a cada presa, el programa troba on és realment el signe (lsc_core.analyse_take) i en retalla
finestres de 2,0 s (100 trames) en moltes posicions. Cada finestra s'etiqueta segons QUANTA PART DEL
SIGNE CONTÉ:

  signe amb moviment (des que la mà surt del repòs fins que hi torna = el "moviment"):
     la finestra conté >= 80 % del moviment   -> el signe
     la finestra conté <= 30 % del moviment   -> repòs (un tros de signe NO és el signe)
     entremig                                 -> no es fa servir (és ambigua)
  lletra (la part quieta és la "lletra mantinguda"):
     >= 1,0 s de lletra mantinguda dins de la finestra   -> la lletra
     <= 0,3 s de lletra mantinguda dins de la finestra   -> repòs
     entremig                                            -> no es fa servir

Així el model aprèn: "un signe COMPLET en qualsevol lloc dels 2 s = el signe; el començament o el final
d'un signe = repòs". En temps real, el guant mira els darrers 2 s cada 200 ms, de manera que, comencis
quan comencis, unes quantes finestres seguides contenen el signe sencer (i lsc_decider.h el diu una
sola vegada) i les finestres que el tallen diuen repòs. Això és el que soluciona el problema de la
primera versió: "he començat al segon 1,9 i les dues finestres han fallat".

Les posicions es reparteixen perquè el signe aparegui al principi, al mig i al final de la finestra, i
les còpies d'entrenament es varien de la mateixa manera que varien les sessions reals:
  * velocitat de 0,85 a 1,15 vegades          * direcció "endavant" +-20 graus (posició de zero / deriva del giroscopi)
  * calibratge dels dits +-10 %               * una mica de soroll
  * signes d'una mà: la mà esquerra es substitueix per dades de la mà esquerra d'altres gravacions,
    perquè el model aprengui a no fer cas d'on és la mà esquerra (el model antic ho havia après).
Les dades de test no es varien mai i no comparteixen cap presa amb les d'entrenament.

Sortida: un CSV per finestra, amb el nom <signe>.<número>.csv (Edge Impulse llegeix l'etiqueta del nom),
i a més report.html (què s'ha fet servir, què s'ha rebutjat i per què, i un gràfic per bloc) i windows.npz.
Necessita: numpy (i scikit-learn per a --check).
"""
import argparse
import glob
import html
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np

import lsc_core as C

W = C.WINDOW                              # trames per finestra (100)
POS_DYN, NEG_DYN = 0.80, 0.30             # signes amb moviment: >= 80 % = signe, <= 30 % = repòs
POS_HOLD, NEG_HOLD = 50, 15               # lletres: trames de lletra mantinguda dins de la finestra (1,0 s / 0,3 s)
LEFT_ACTIVE = 1.5                         # moviment de la mà esquerra a partir del qual es considera que "s'ha mogut"


# Lectura i preses
# Llegeix totes les gravacions, calcula les 34 dades d'entrada de cada trama i, en els blocs de
# preses, troba on és cada signe.
def load_recordings(raw_dir, signs, rest_key):
    recs = []
    for js in sorted(glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)):
        try:
            meta, raw = C.load_take(js)
        except (OSError, ValueError) as e:
            print("cannot read", js, e)
            continue
        if raw.shape[1] != len(C.RAW_COLUMNS):
            print("wrong columns, skipped:", js)
            continue
        label = meta.get("label")
        if label != rest_key and label not in signs:            # signes que ja no són al vocabulari
            print("label '%s' is not in signs.json, skipped: %s" % (label, js))
            continue
        feat = C.features(raw, meta["calibration"])             # amb el calibratge de la seva sessió
        valid = C.frames_valid(raw)
        rec = {"path": js, "meta": meta, "feat": feat, "valid": valid, "label": label,
               "session": meta.get("session", "?"), "segments": [], "rejected": [], "exclude": []}
        if meta.get("kind") == "block":
            find_segments(rec, signs[label])
        recs.append(rec)
    return recs


def find_segments(rec, sign):
    """Torna a analitzar cada presa d'un bloc (amb la mateixa funció que va fer servir l'aplicació en gravar)."""
    meta, feat = rec["meta"], rec["feat"]
    N = len(feat)
    cues = sorted(meta.get("cues", []))                             # senyals "ARA!" de les preses bones
    allcues = sorted(cues + [b["cue"] for b in meta.get("bad", [])])
    perform = int(meta.get("perform_s", 2.5) * C.FS)
    for b in meta.get("bad", []):                                  # preses rebutjades en gravar (per tu o pel programa):
        rec["exclude"].append((b["cue"] - 15, min(N, b["cue"] + perform + C.FS)))   # aquest tros no es fa servir
        rec["rejected"].append(b.get("reason", "?"))
    for cue in cues:
        nxt = [c for c in allcues if c > cue]
        a = max(0, cue - int(1.6 * C.FS))                          # de 1,6 s abans del senyal...
        b = min(N, (nxt[0] - int(1.0 * C.FS)) if nxt else N)       # ...fins a 1 s abans del senyal següent
        res = C.analyse_take(feat[a:b], cue - a, sign)
        if not res["ok"]:
            rec["rejected"].append(res["reason"])
            rec["exclude"].append((cue - 15, min(N, cue + perform + C.FS)))
            continue
        # Presa bona: on comença i acaba el moviment i, si és una lletra, la part mantinguda.
        seg = {"cue": cue, "start": res["start"] + a, "end": res["end"] + a, "static": sign["type"] == "static",
               "hold": (res["hold_start"] + a, res["hold_end"] + a) if res["hold_start"] is not None else None}
        el = C.motion_energy(feat, "l")[seg["start"]:seg["end"]]
        seg["left_moves"] = bool(len(el) and el.max() > LEFT_ACTIVE)   # s'ha mogut la mà esquerra?
        rec["segments"].append(seg)


# Quant se solapen dos intervals [a0, a1) i [b0, b1).
def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_label(rec, w0, span):
    """Etiqueta d'un tros de la gravació [w0, w0+span): ('pos', presa) = el signe, ('neg', toca un signe?) = repòs,
    o None = no es fa servir."""
    w1 = w0 + span
    for a, b in rec["exclude"]:                                    # toca una presa rebutjada
        if overlap(w0, w1, a, b) > 0:
            return None
    i0, i1 = int(math.floor(w0)), int(math.ceil(w1))
    if i0 < 0 or i1 > len(rec["feat"]) or not rec["valid"][i0:i1].all():   # surt de la gravació o té dades perdudes
        return None
    pos, touches, ambiguous = [], False, False
    for s in rec["segments"]:
        if s["static"]:
            ho = overlap(w0, w1, *s["hold"]) * W / span                  # trames de lletra mantinguda dins la finestra
            if ho >= POS_HOLD:
                pos.append(s)
            elif ho > NEG_HOLD:
                ambiguous = True
        else:
            cov = overlap(w0, w1, s["start"], s["end"]) / (s["end"] - s["start"])   # part del moviment dins la finestra
            if cov >= POS_DYN:
                pos.append(s)
            elif cov > NEG_DYN:
                ambiguous = True
        if overlap(w0, w1, s["start"], s["end"]) > 0:
            touches = True                                               # conté almenys un tros d'un signe
    if ambiguous or len(pos) > 1:
        return None
    if pos:
        return ("pos", pos[0])
    return ("neg", touches)


def extract(feat, w0, r=1.0):
    """Retalla W trames a partir de la trama w0, agafant una trama cada r trames de l'original
    (r diferent d'1 = signe fet més ràpid o més lent)."""
    if r == 1.0 and float(w0).is_integer():
        return feat[int(w0): int(w0) + W].copy()
    t = w0 + np.arange(W) * r
    i = np.clip(np.floor(t).astype(int), 0, len(feat) - 2)
    u = (t - i)[:, None]
    return feat[i] * (1 - u) + feat[i + 1] * u                     # interpolació entre dues trames veïnes


# Variació de les dades (augment)
# Gira la direcció "endavant" d'una mà 'deg' graus al voltant de l'eix vertical (simula una posició
# de zero diferent o la deriva del giroscopi).
def rotate_heading(x, hand, deg):
    off = 0 if hand == "r" else 17
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    for a in (5, 8, 14):                                                # eix X, eix Z i acceleració
        u, v = x[:, off + a].copy(), x[:, off + a + 1].copy()
        x[:, off + a], x[:, off + a + 1] = c * u - s * v, s * u + c * v


# Crea una còpia variada d'una finestra
def augment(x, rng, left_pool=None, one_hand=False):
    x = x.copy()
    for hand in ("r", "l"):
        rotate_heading(x, hand, rng.uniform(-20, 20))
        off = 0 if hand == "r" else 17
        g, o = rng.uniform(0.9, 1.1, 5), rng.uniform(-0.05, 0.05, 5)     # calibratge dels dits +-10 % i petit desplaçament
        x[:, off:off + 5] = np.clip(x[:, off:off + 5] * g + o, C.FLEX_MIN, C.FLEX_MAX)
    x += rng.normal(0, 0.01, x.shape)                                     # soroll petit
    if one_hand and left_pool and rng.random() < 0.5:
        x[:, 17:] = left_pool[rng.integers(len(left_pool))]
    return x


# Separació i finestres
def split_point(rec, test_session, holdout):
    """Trama a partir de la qual la gravació va al test (len = tot entrenament, 0 = tot test).
    Per defecte, de cada bloc es reserven per al test les darreres preses (el 20 %)."""
    N = len(rec["feat"])
    if test_session:
        return 0 if rec["session"] == test_session else N
    if holdout <= 0:
        return N
    if rec["meta"].get("kind") == "block":
        segs = rec["segments"]
        if len(segs) < 3:
            return N
        k = max(1, int(round(len(segs) * holdout)))
        return segs[-k]["cue"] - int(1.6 * C.FS)
    return int(N * (1 - holdout))


def windows_for(rec, signs, rest_key, rng, args, left_pool):
    """Totes les finestres d'una gravació. Torna {'train': [(finestra, etiqueta, tipus)], 'test': [...]}."""
    out = {"train": [], "test": []}
    N = len(rec["feat"])
    cut = split_point(rec, args.test_session, args.holdout)
    feat = rec["feat"]

    # A quina part va un tros: entrenament, test, o cap (si queda partit pel mig).
    def part(w0, span):
        if w0 + span <= cut:
            return "train"
        if w0 >= cut:
            return "test"
        return None

    # Gravacions de repòs: una finestra cada rest_stride trames (i, a l'entrenament, la meitat amb una còpia variada).
    if rec["meta"].get("kind") == "rest":
        for w0 in range(0, N - W + 1, args.rest_stride):
            lab = window_label(rec, w0, W)
            p = part(w0, W)
            if lab and p:
                x = extract(feat, w0)
                out[p].append((x, rest_key, "rest"))
                if p == "train" and rng.random() < 0.5:
                    out[p].append((augment(x, rng), rest_key, "rest+aug"))
        return out

    #es classifiquen totes les posicions possibles (cada 2 trames).
    label = rec["label"]
    one_hand = int(signs[label].get("hands", 1)) == 1
    by_seg = defaultdict(list)                              # posicions que contenen cada presa
    negs = []                                               # posicions que són repòs
    for w0 in range(0, N - W + 1, 2):
        lab = window_label(rec, w0, W)
        if lab is None:
            continue
        if lab[0] == "pos":
            by_seg[lab[1]["cue"]].append(w0)
        else:
            negs.append((w0, lab[1]))
    for seg in rec["segments"]:
        ws = by_seg.get(seg["cue"], [])
        if not ws:
            continue
        p = part(min(ws), W + max(ws) - min(ws))
        if p is None:                                       # les posicions queden a banda i banda de la separació, es queda un costat
            ws = [w for w in ws if part(w, W)]
            if not ws:
                continue
            p = part(ws[0], W)
            ws = [w for w in ws if part(w, W) == p]
        # Finestres sense variar, repartides perquè el signe quedi al principi, al mig i al final de la finestra.
        k = args.pos_plain if p == "train" else args.pos_test
        picks = sorted(set(ws[int(round(i * (len(ws) - 1) / max(1, k - 1)))] for i in range(k)))
        for w0 in picks:
            out[p].append((extract(feat, w0), label, "plain"))
        if p != "train":
            continue
        # Còpies variades (només a l'entrenament): posició i velocitat a l'atzar, més les variacions d'augment().
        for _ in range(args.pos_aug):
            for _try in range(20):
                r = rng.uniform(0.85, 1.15)
                w0 = rng.uniform(max(0, seg["start"] - W * r), min(N - W * r - 1, seg["end"]))
                lab = window_label(rec, w0, W * r)
                if lab and lab[0] == "pos" and lab[1]["cue"] == seg["cue"] and part(w0, W * r) == "train":
                    out["train"].append((augment(extract(feat, w0, r), rng, left_pool, one_hand), label, "aug"))
                    break
    # Exemples de repòs del mateix bloc: primer els trossos de signe (els més difícils) i després repòs quiet.
    partial = [w for w, t in negs if t]
    quiet = [w for w, t in negs if not t]
    rng.shuffle(partial)
    rng.shuffle(quiet)
    n_seg = max(1, len(rec["segments"]))
    for w0 in partial[: args.neg_partial * n_seg] + quiet[: n_seg]:
        p = part(w0, W)
        if p:
            x = extract(feat, w0)
            out[p].append((x if p == "test" else augment(x, rng), rest_key, "fragment" if w0 in partial else "block-rest"))
    return out


# Recull fins a n finestres de la mà esquerra en repòs (de les gravacions de repòs), per substituir la
# mà esquerra en els signes d'una mà.
def left_pool_from(recs, rng, n=400):
    pool = []
    for rec in recs:
        if rec["meta"].get("kind") != "rest":
            continue
        N = len(rec["feat"])
        for w0 in range(0, N - W + 1, 25):
            if rec["valid"][w0:w0 + W].all():
                pool.append(rec["feat"][w0:w0 + W, 17:].copy())   #ccolumnes 17-33: mà esquerra
    rng.shuffle(pool)
    return pool[:n]


# Fitxers de sortida
# Escriu cada finestra en un CSV amb el format que llegeix Edge Impulse (temps en ms + 34 columnes).
def write_ei(folder, items):
    os.makedirs(folder, exist_ok=True)
    header = "timestamp," + ",".join(C.FEAT_COLUMNS) + "\n"
    for i, (x, label, info) in enumerate(items):
        with open(os.path.join(folder, "%s.w%05d.csv" % (label, i)), "w", encoding="utf-8", newline="") as fh:
            fh.write(header)
            for k in range(W):
                fh.write(str(k * C.FRAME_MS) + "," + ",".join("%.4f" % v for v in x[k]) + "\n")


def summarise(X):
    """Resum d'una finestra amb una mida fixa, per al model ràpid de --check: mitjanes de 10 trossos,
    desviació típica i amplitud de cada columna."""
    X = np.asarray(X)
    pooled = X.reshape(len(X), 10, W // 10, X.shape[2]).mean(axis=2).reshape(len(X), -1)
    return np.concatenate([pooled, X.std(axis=1), X.max(axis=1) - X.min(axis=1)], axis=1)


# Model ràpid (un bosc d'arbres de decisió) que només serveix per veure aquí, abans d'Edge Impulse, què es confon.
def quick_model():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, n_jobs=-1, random_state=0)


# Entrena el model ràpid i torna l'exactitud, els encerts per classe i les confusions més freqüents.
def check_model(Xtr, ytr, Xte, yte, labels):
    clf = quick_model()
    clf.fit(summarise(Xtr), ytr)
    pred = clf.predict(summarise(Xte))
    acc = float(np.mean(pred == np.asarray(yte)))
    cm = Counter(zip(yte, pred))
    per = {l: (cm[(l, l)], sum(v for (a, b), v in cm.items() if a == l)) for l in labels}
    confusions = sorted(((v, a, b) for (a, b), v in cm.items() if a != b), reverse=True)[:12]
    return clf, acc, per, confusions


# Informe
# Gràfic d'un bloc: çmoviment de la mà dreta (línia blanca), signes trobats (verd), senyals "ARA!" (blau)
# i preses rebutjades (vermell).
def svg_block(rec):
    feat = rec["feat"]
    e = C.motion_energy(feat, "r")
    N = len(e)
    Wpx, Hpx = 900, 70
    sx = Wpx / max(1, N)
    top = max(4.0, float(np.percentile(e, 99)))
    pts = " ".join("%.1f,%.1f" % (i * sx, Hpx - min(1.0, e[i] / top) * (Hpx - 4)) for i in range(0, N, 2))
    parts = []
    for s in rec["segments"]:
        parts.append('<rect x="%.1f" width="%.1f" y="0" height="%d" fill="#3ccf7a33"/>' % (s["start"] * sx, (s["end"] - s["start"]) * sx, Hpx))
        if s["hold"]:
            parts.append('<rect x="%.1f" width="%.1f" y="%d" height="6" fill="#3ccf7a"/>' % (s["hold"][0] * sx, (s["hold"][1] - s["hold"][0]) * sx, Hpx - 6))
        parts.append('<line x1="%.1f" x2="%.1f" y1="0" y2="%d" stroke="#5aa9ff"/>' % (s["cue"] * sx, s["cue"] * sx, Hpx))
    for a, b in rec["exclude"]:
        parts.append('<rect x="%.1f" width="%.1f" y="0" height="%d" fill="#ff5d5d33"/>' % (a * sx, (b - a) * sx, Hpx))
    return ('<svg width="%d" height="%d" style="background:#0c0f13">%s<polyline points="%s" fill="none" stroke="#e9edf2" stroke-width="1"/></svg>'
            % (Wpx, Hpx, "".join(parts), pts))


# Escriu report.html, una taula per signe (preses fetes servir, durada mitjana, si es mou la mà esquerra,
# finestres i motius de rebuig), els avisos, el resultat de --check i el gràfic de cada bloc.
def write_report(path, recs, signs, rest_key, counts, check, warnings):
    e = html.escape
    rows = []
    for key, s in list(signs.items()) + [(rest_key, {"text": "Repòs", "type": "-"})]:
        rs = [r for r in recs if r["label"] == key]
        segs = [g for r in rs for g in r["segments"]]
        rej = Counter(x for r in rs for x in r["rejected"])
        dur = np.mean([(g["end"] - g["start"]) / C.FS for g in segs]) if segs else 0
        lm = np.mean([g["left_moves"] for g in segs]) if segs else 0
        secs = sum(len(r["feat"]) for r in rs if r["meta"].get("kind") == "rest") / C.FS
        rows.append("<tr><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%d / %d</td><td>%s</td></tr>" % (
            e(s["text"]), s["type"], len(rs), len(segs) if key != rest_key else "%.0f s" % secs,
            "%.1f s" % dur if segs else "", "%.0f %%" % (100 * lm) if segs else "",
            counts["train"].get(key, 0), counts["test"].get(key, 0),
            e("; ".join("%s (%d)" % (k, v) for k, v in rej.most_common()))))
    chk = ""
    if check:
        acc, per, conf = check
        chk = "<h2>Quick model trained here (stand-in for Edge Impulse)</h2><p>Test windows correct: <b>%.1f %%</b></p><table><tr><th>Class</th><th>Correct / test windows</th></tr>%s</table><p>Most frequent confusions (true → predicted, windows):</p><ul>%s</ul>" % (
            100 * acc, "".join("<tr><td>%s</td><td>%d / %d</td></tr>" % (e(k), a, b) for k, (a, b) in per.items()),
            "".join("<li>%s → %s: %d</li>" % (e(a), e(b), v) for v, a, b in conf) or "<li>none</li>")
    blocks = "".join("<p><b>%s</b> · %s · %d takes used, %d rejected<br>%s</p>" % (
        e(r["label"]), e(os.path.basename(r["path"])), len(r["segments"]), len(r["rejected"]), svg_block(r))
        for r in recs if r["meta"].get("kind") == "block")
    page = """<!doctype html><meta charset="utf-8"><title>Dataset report</title>
<style>body{background:#111418;color:#e9edf2;font:14px system-ui;margin:20px}td,th{border-bottom:1px solid #2c333d;padding:4px 8px;text-align:left}
table{border-collapse:collapse}.w{color:#ffb547}</style>
<h1>LSC dataset report</h1>%s
<table><tr><th>Sign</th><th>Type</th><th>Files</th><th>Takes used</th><th>Mean movement</th><th>Left hand moves</th><th>Windows train / test</th><th>Rejected takes</th></tr>%s</table>
%s<h2>Blocks</h2><p>White: right-hand movement. Green: the sign found (bar = held part of a letter). Blue: the "ARA!" cue. Red: rejected take (not used).</p>%s""" % (
        "".join('<p class="w">⚠ %s</p>' % e(w) for w in warnings), "".join(rows), chk, blocks)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


# Programa principal
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Opcions que es poden canviar des de la línia d'ordres (entre parèntesis, el valor per defecte).
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))                 # carpeta de les gravacions
    ap.add_argument("--out", default=os.path.join(here, "dataset", "edge_impulse"))       # carpeta de sortida
    ap.add_argument("--test-session", default=None, help="put this whole session in the test set")   # sessió sencera per al test
    ap.add_argument("--holdout", type=float, default=0.2, help="last part of every recording used for testing")   # part final de cada gravació per al test (20 %)
    ap.add_argument("--pos-plain", type=int, default=3, help="windows per take (sign at start/middle/end)")       # finestres sense variar per presa
    ap.add_argument("--pos-aug", type=int, default=4, help="extra varied windows per take")                       # còpies variades per presa
    ap.add_argument("--pos-test", type=int, default=3)                                                            # finestres de test per presa
    ap.add_argument("--neg-partial", type=int, default=2, help="sign-fragment windows (repos) per take")           # trossos de signe (repòs) per presa
    ap.add_argument("--rest-stride", type=int, default=25, help="frames between repos windows in rest recordings")  # separació entre finestres de repòs
    ap.add_argument("--rest-ratio", type=float, default=1.5, help="max repos windows / mean windows of a sign")     # màxim de repòs respecte d'un signe
    ap.add_argument("--check", action="store_true", help="train a quick model here and report confusions")
    ap.add_argument("--seed", type=int, default=1)                                     # llavor de l'atzar (resultats repetibles)
    args = ap.parse_args()

    data, signs = C.load_signs()
    rest_key = data.get("rest_key", "repos")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    recs = load_recordings(args.raw, signs, rest_key)
    if not recs:
        print("No recordings found in", args.raw)
        return 1

    # Avisos: signes sense preses, mans mal definides a signs.json, poques preses, falta de repòs o d'altres sessions.
    warnings = []
    for key, s in signs.items():
        segs = [g for r in recs if r["label"] == key for g in r["segments"]]
        if not segs:
            warnings.append("%s: no usable takes yet" % s["text"])
            continue
        lm = np.mean([g["left_moves"] for g in segs])
        if int(s.get("hands", 1)) == 1 and lm > 0.6:
            warnings.append("%s is marked one-hand but the left hand moves in %.0f %% of the takes: if it is a two-hand sign set \"hands\": 2 in signs.json" % (s["text"], 100 * lm))
        if int(s.get("hands", 1)) == 2 and lm < 0.2:
            warnings.append("%s is marked two-hand but the left hand barely moves (%.0f %% of takes)" % (s["text"], 100 * lm))
        if len(segs) < 20:
            warnings.append("%s: only %d takes (aim for %d, from at least 2-3 sessions)" % (s["text"], len(segs), data.get("takes_target", 40)))
    if not any(r["label"] == rest_key for r in recs):
        warnings.append("No 'repos' recordings: record at least 3-5 minutes (button Repòs).")
    sessions = sorted(set(r["session"] for r in recs))
    if len(sessions) < 2:
        warnings.append("All data comes from ONE session. Record on different days / after taking the gloves off, ideally both of you.")
    if args.test_session and args.test_session not in sessions:
        print("Unknown session %s. Sessions: %s" % (args.test_session, ", ".join(sessions)))
        return 1

    # Genera les finestres de totes les gravacions.
    pool = left_pool_from(recs, rng)
    split = {"train": [], "test": []}
    for rec in recs:
        w = windows_for(rec, signs, rest_key, rng, args, pool)
        for p in split:
            split[p] += w[p]

    #a l'entrenament no hi pot haver massa finestres de repòs (com a màxim 1,5 vegades la
    # mitjana d'un signe). Es guarden la meitat de trossos de signe i la resta de repòs normal.
    per = Counter(l for _, l, _ in split["train"])
    sign_counts = [per[k] for k in signs if per[k]]
    if sign_counts and per[rest_key] > args.rest_ratio * np.mean(sign_counts):
        cap = int(args.rest_ratio * np.mean(sign_counts))
        rest = [it for it in split["train"] if it[1] == rest_key]
        frag = [it for it in rest if it[2] == "fragment"]
        other = [it for it in rest if it[2] != "fragment"]
        rng.shuffle(frag)
        rng.shuffle(other)
        keep_f = frag[: cap // 2]
        keep = keep_f + other[: cap - len(keep_f)]
        split["train"] = [it for it in split["train"] if it[1] != rest_key] + keep

    # Escriu les carpetes training i testing (esborrant les anteriors), windows.npz i dataset_info.json.
    for p in ("training", "testing"):
        shutil.rmtree(os.path.join(args.out, p), ignore_errors=True)
    rng.shuffle(split["train"])
    write_ei(os.path.join(args.out, "training"), split["train"])
    write_ei(os.path.join(args.out, "testing"), split["test"])
    counts = {p: Counter(l for _, l, _ in split[p]) for p in split}
    labels = list(signs) + [rest_key]
    np.savez_compressed(os.path.join(args.out, "windows.npz"),
                        Xtr=np.array([x for x, _, _ in split["train"]], dtype=np.float32),
                        ytr=np.array([l for _, l, _ in split["train"]]),
                        Xte=np.array([x for x, _, _ in split["test"]], dtype=np.float32),
                        yte=np.array([l for _, l, _ in split["test"]]), labels=np.array(labels))
    with open(os.path.join(args.out, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"axes": C.FEAT_COLUMNS, "window_ms": W * C.FRAME_MS, "frequency_hz": C.FS, "labels": labels,
                   "train": counts["train"], "test": counts["test"], "sessions": sessions}, fh, indent=1)

    # Resum a la pantalla: finestres per signe i motius de rebuig.
    print("\nWindows        train   test")
    for k in labels:
        print("  %-12s %6d %6d" % (k, counts["train"].get(k, 0), counts["test"].get(k, 0)))
    rejected = Counter(x for r in recs for x in r["rejected"])
    if rejected:
        print("\nRejected takes:")
        for k, v in rejected.most_common():
            print("  %3d  %s" % (v, k))
    # Opció --check: model ràpid entrenat aquí.
    check = None
    if args.check:
        if not split["test"]:
            print("\n--check needs test windows (record more takes)")
        else:
            Xtr = np.array([x for x, _, _ in split["train"]]); ytr = [l for _, l, _ in split["train"]]
            Xte = np.array([x for x, _, _ in split["test"]]); yte = [l for _, l, _ in split["test"]]
            _, acc, per_c, conf = check_model(Xtr, ytr, Xte, yte, labels)
            check = (acc, per_c, conf)
            print("\nQuick model (stand-in for Edge Impulse): %.1f %% of test windows correct" % (100 * acc))
            for v, a, b in conf[:8]:
                print("  %-12s -> %-12s %d" % (a, b, v))
    for w in warnings:
        print("WARNING:", w)
    report = os.path.join(os.path.dirname(args.out.rstrip("/\\")), "report.html")
    write_report(report, recs, signs, rest_key, counts, check, warnings)
    print("\nEdge Impulse files: %s\nReport: %s" % (args.out, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
