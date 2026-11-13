#
# lsc_app.py  -  aplicació de l'ordinador per fer servir els guants.
#
#     python lsc_app.py            amb els guants connectats per USB
#     python lsc_app.py --demo     sense guants: una persona i uns guants simulats (lsc_sim.py)
#
# Què fa:
#   * Busca els guants pels ports USB i els reconeix pel missatge de presentació ("hello").
#   * Reenvia al guant dret cada línia del guant esquerre (L,...), perquè el dret tingui les dues mans.
#   * Rep del guant dret les trames (línies F, 50 per segon) i, cada 200 ms, la sortida del model
#     i la decisió (línies P).
#   * Obre una pàgina web (lsc_app.html) amb quatre pestanyes, i s'hi comunica per WebSocket:
#       1. Calibrar: posició de zero (direcció "endavant") i 5 repeticions de mà oberta / puny tancat.
#       2. Gravar:   enregistrament guiat de preses de cada signe i de repòs (les dades d'entrenament).
#       3. Provar:   demana signes a l'atzar i compta encerts, errors, signes no detectats i falses alarmes.
#       4. Traduir:  construeix la frase amb els signes que diu el guant (lsc_frase.py), la mostra
#                    a la pantalla i la llegeix en veu alta.
#
import argparse
import asyncio
import functools
import glob
import http.server
import json
import os
import random
import socketserver
import threading
import time
import webbrowser
from collections import deque

import numpy as np

import lsc_core as C
from lsc_frase import Frase

HERE = os.path.dirname(os.path.abspath(__file__))
HTTP_PORT, WS_PORT, BAUD = 8000, 8765, 460800    # port de la pàgina web, port del WebSocket i velocitat USB
REST_S = 2.0                   # repòs abans de la primera presa
CHECK_S = 1.0                  # després del temps de l'"ARA!": s'espera això i es comprova la presa
NEXT_S = 1.5                   # i després es reposa això (amb el resultat a la pantalla) abans del següent "ARA!"
PERFORM_S = {"dynamic": 2.5, "static": 3.5}      # temps per fer el signe (amb moviment / lletra)
PRE_S = 1.6                    # repòs abans del senyal que es fa servir per analitzar una presa
try:                                                          # canvia quan canvia lsc_app.html
    import zlib
    PAGE_VERSION = "%08x" % zlib.crc32(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsc_app.html"), "rb").read())
except OSError:
    PAGE_VERSION = "?"
SENTENCE_END_S = 2.4           # Traduir: aquest temps de "repòs" (finestres del model) tanca la frase.
                               # Dins d'una frase, entre paraula i paraula, el model ja diu repòs fins a uns
                               # 1,2 s (mesurat amb gravacions enganxades), per això ha de ser més llarg.
DECISION_DEFAULT = {"t_on": 0.70, "t_rel": 0.50, "hold": 5, "delay": 2}   # = valors per defecte del guant
COL = {n: i for i, n in enumerate(C.RAW_COLUMNS)}            # posició de cada columna d'una trama


# Connexions USB
class SerialLink:
    """Un port USB. Llegeix les línies en un fil a part; el paper (esquerre/dret) surt de la presentació del guant."""

    def __init__(self, hub, port):
        import serial
        self.hub, self.port, self.role = hub, port, None
        self.ser = serial.Serial(port, BAUD, timeout=0.2)
        self.lock = threading.Lock()
        self.alive = True
        self.opened = time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    # Envia text al guant (protegit perquè dos fils no escriguin alhora).
    def write(self, s):
        try:
            with self.lock:
                self.ser.write(s.encode("ascii"))
        except Exception:
            self.close()

    def close(self):
        self.alive = False
        try:
            self.ser.close()
        except Exception:
            pass
#talla el que arriba en línies i les passa al Hub. Si el guant encara no s'ha
    # presentat, li pregunta "qui ets?" ("?").
    def _reader(self):
        buf = b""
        while self.alive:
            try:
                chunk = self.ser.read(4096)
            except Exception:
                break
            if not chunk:
                if self.role is None and time.time() - self.opened > 1.0:
                    self.write("?\n")
                continue
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.decode("ascii", "replace").strip()
                if line:
                    self.hub.on_line(self, line)
        self.alive = False
        self.hub.on_link_closed(self)


# Cada 2 s mira quins ports USB hi ha i obre els nous. Tanca els que en 6 s no s'han presentat.
def port_scanner(hub):
    import serial.tools.list_ports
    tried = {}
    while True:
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        for dev in ports:
            if dev in hub.links or time.time() - tried.get(dev, 0) < 8:
                continue
            tried[dev] = time.time()
            try:
                hub.links[dev] = SerialLink(hub, dev)
            except Exception:
                pass
        for dev, link in list(hub.links.items()):          # ports que no s'han presentat mai
            if link.role is None and time.time() - link.opened > 6:
                link.close()
                hub.links.pop(dev, None)
        time.sleep(2)


# Guants simulats (--demo)
class DemoGloves:
    """Persona i guants simulats. Parla el mateix protocol que el guant dret real."""

    def __init__(self, hub):
        from lsc_sim import Person
        self.hub = hub
        self.person = Person(seed=int(time.time()) % 1000)
        self.role = "demo"
        self.port = "DEMO"
        self.labels = list(hub.signs) + [hub.rest_key]
        self.pending = None
        self.autoplay = False
        self.performing = None

    # Respon a les ordres de l'aplicació com ho faria el guant dret (presentació, calibratge, llindars).
    def write(self, s):
        s = s.strip()
        if s.startswith("?"):
            self._hello()
        elif s.startswith("C,") and C.unseal(s):
            self.hub.on_line(self, json.dumps({"type": "cal_ok"}))
        elif s.startswith("T,") and C.unseal(s):
            v = C.unseal(s).split(",")
            msg = {"type": "thr_ok", "t_on": float(v[1]), "t_rel": float(v[2])}
            if len(v) >= 5:
                msg.update(hold=int(v[3]), delay=int(v[4]))
            self.hub.on_line(self, json.dumps(msg))

    def _hello(self):
        self.hub.on_line(self, json.dumps({"type": "hello", "hand": "left", "fw": "2.0", "imu": 1, "bias_ok": 1}))
        self.hub.on_line(self, json.dumps({"type": "hello", "hand": "right", "fw": "2.0", "imu": 1, "bias_ok": 1,
                                           "axes": 34, "window": 100, "labels": self.labels,
                                           "rest_index": len(self.labels) - 1}))

    # La persona simulada fa un signe d'aquí a 'delay' segons.
    def perform(self, key, delay=0.0, hold=2.3):
        async def go():
            await asyncio.sleep(delay)
            self.person.sign(key, hold=hold)
            self.performing = key
        asyncio.get_event_loop().create_task(go())

    # Genera 50 trames per segon i, cada 10 trames, una línia P. En la simulació el "model" diu
    # el signe quan la persona l'acaba (no hi ha cap xarxa neuronal).
    async def run(self):
        self._hello()
        nxt = time.perf_counter()
        n = 0
        last_auto = time.time()
        while True:
            row = self.person.frame()
            r, l = row[2:18], row[18:34]
            fmt = lambda h: ",".join(C._fmt(v) for v in h)
            self.hub.on_line(self, C.seal("F,%d,%s,%s,%d" % (row[1], fmt(r), fmt(l), row[-1])))
            n += 1
            if self.performing and not self.person.busy():
                self.pending, self.performing = self.performing, None
            if n % 10 == 0:
                p = [0.0] * len(self.labels)
                emit = -1
                if self.pending:
                    emit = self.labels.index(self.pending)
                    p[emit] = 0.93
                    self.pending = None
                else:
                    p[-1] = 0.95
                rest = (1.0 - sum(p)) / len(p)
                p = [v + rest for v in p]
                self.hub.on_line(self, C.seal("P,%d,%s,%d" % (row[1], ",".join("%.3f" % v for v in p), emit)))
            # A la pantalla de traducció, la persona simulada fa signes a l'atzar cada 3,5 s.
            if self.autoplay and not self.person.busy() and not self.performing and time.time() - last_auto > 3.5:
                last_auto = time.time()
                self.perform(random.choice(list(self.hub.signs)))
            nxt += 0.02
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))


#el centre de l'aplicació
class Hub:
    # Estat de tota l'aplicació: guants connectats, calibratge, sessió, activitat en curs i frase en construcció.
    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.abspath(args.data)
        self.signs_data, self.signs = C.load_signs()
        self.rest_key = self.signs_data.get("rest_key", "repos")
        self.loop = None
        self.clients = set()                               # pàgines web connectades
        self.links = {}                                    # ports USB oberts
        self.boards = {"left": None, "right": None}       # connexió de cada guant
        self.hello = {"left": None, "right": None}        # última presentació de cada guant
        self.rstatus = {}                                  # últim missatge d'estat del guant dret
        self.frames = deque(maxlen=C.FS * 30)              # darrers 30 s de trames
        self.listeners = []                                # llistes que estan gravant trames
        self.last_seq = None
        self.frame_count = 0
        self.fps = 0.0
        self.cal = None
        self.cal_sent = False
        self.session = None
        self.person = args.person
        self.task = None
        self.activity = "idle"
        self.stop_flag = False
        self.discard_flag = False
        self.emits = deque(maxlen=200)                     # signes dits pel guant, amb l'hora
        self.labels = []                                   # classes del model (de la presentació del guant dret)
        self.demo = None
        self.counts = {}
        self.cal_waiter = None
        self.thresholds = None             # ajustos de decisió triats a la pestanya Provar {t_on, t_rel, hold, delay}
        self.thresholds_now = None         # els que diu que té el guant
        # Traduir: la frase en construcció
        self.frase = Frase({k: v.get("text", k) for k, v in self.signs.items()})
        self.frase_words = []              # últimes paraules enviades a la pantalla
        self.spoken = []                   # paraules ja dites en veu alta en aquesta frase
        self.rest_run = 0                  # finestres seguides en què el model ha dit repòs
        self.sentence_end_windows = int(round(SENTENCE_END_S / 0.2))
        self.reread = True

    # Línies que arriben (de qualsevol fil)
    def on_line(self, link, line):
        # Les línies del guant esquerre es reenvien al dret immediatament, sense esperar.
        if line.startswith("L,"):
            if link.role is None:
                link.role = "left"
            right = self.boards.get("right")
            if right is not None and right is not link:
                right.write(line + "\n")                      # reenviament al guant dret
            return
        # La resta es processa al fil principal.
        if self.loop:
            self.loop.call_soon_threadsafe(self._on_line_loop, link, line)

    def on_link_closed(self, link):
        if self.loop:
            self.loop.call_soon_threadsafe(self._closed_loop, link)

    # Un guant s'ha desconnectat: s'oblida.
    def _closed_loop(self, link):
        for role, l in self.boards.items():
            if l is link:
                self.boards[role] = None
                self.hello[role] = None
                if role == "right":
                    self.cal_sent = False
        self.links.pop(link.port, None)

    # Processa una línia del guant dret (F, P o JSON).
    def _on_line_loop(self, link, line):
        # Línia F (trama): es guarda. Si se n'ha perdut alguna, s'omple amb l'anterior marcada com a no vàlida.
        if line.startswith("F,"):
            v = C.parse_frame(line)
            if v is None:
                return
            seq = int(v[0])
            if self.last_seq is not None and 0 < seq - self.last_seq <= 10:
                for missing in range(self.last_seq + 1, seq):       # línia perduda: es repeteix, marcada com a no vàlida
                    row = list(self.frames[-1]) if self.frames else [0.0] * len(C.RAW_COLUMNS)
                    row[0], row[1], row[-1] = missing * C.FRAME_MS, missing, 65535
                    self._push_row(row)
            self.last_seq = seq
            self._push_row([seq * C.FRAME_MS] + v)
            return
        # Línia P (sortida del model i decisió del guant, cada 200 ms).
        if line.startswith("P,"):
            p = C.parse_probs(line)
            if p is None:
                return
            seq, probs, emit = p
            self.latest_probs = probs
            if emit >= 0 and emit < len(self.labels):
                # El guant ha dit un signe.
                key = self.labels[emit]
                self.emits.append((time.time(), key))
                if self.activity != "translate":           # mentre es tradueix, només s'envia la frase
                    self.broadcast({"type": "emit", "key": key, "text": self.text_of(key), "p": probs[emit]})
                if self.activity == "translate" and key != self.rest_key:
                    self.rest_run = 0
                    self.frase.add(key)                         # s'afegeix a la frase i s'envia a la pantalla
                    self.send_sentence()
            elif self.activity == "translate":
                # Cap signe: es compten les finestres seguides de repòs (probabilitat >= 0,5). Quan n'hi ha
                # prou (2,4 s), la frase s'acaba.
                ri = self.labels.index(self.rest_key) if self.rest_key in self.labels else -1
                top_i = max(range(len(probs)), key=lambda i: probs[i]) if probs else -1
                self.rest_run = self.rest_run + 1 if (top_i == ri and probs[ri] >= 0.5) else 0
                if not self.frase.empty() and self.rest_run >= self.sentence_end_windows:
                    self.end_sentence()
            # Les tres classes més probables, per mostrar-les a la pantalla si es vol.
            self._probs_counter = getattr(self, "_probs_counter", 0) + 1
            top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
            self.broadcast({"type": "probs", "top": [[self.labels[i] if i < len(self.labels) else str(i), probs[i]] for i in top]})
            return
        # Missatges JSON, presentació, ajustos confirmats, estat i confirmació del calibratge.
        if line.startswith("{"):
            try:
                msg = json.loads(line)
            except ValueError:
                return
            t = msg.get("type")
            if t == "hello" and msg.get("hand") in ("left", "right"):
                role = msg["hand"]
                if link.role in (None, role) or link.role == "demo":
                    if link.role != "demo":
                        link.role = role
                    self.boards[role] = link
                    self.hello[role] = msg
                    if role == "right":
                        self.labels = msg.get("labels", [])
                        self.cal_sent = False
                        self.thresholds_now = {k: msg.get(k) for k in ("t_on", "t_rel", "hold", "delay")}
                        if self.cal:
                            self.send_cal()                     # el guant s'ha reiniciat: se li torna a enviar el calibratge
                        if self.thresholds:
                            self.send_thresholds()
            elif t == "thr_ok":
                self.thresholds_now = {k: msg.get(k) for k in ("t_on", "t_rel", "hold", "delay")}
            elif t == "status":
                self.rstatus = msg
            elif t == "cal_ok":
                self.cal_sent = True
                if self.cal_waiter and not self.cal_waiter.done():
                    self.cal_waiter.set_result(True)

    # Guarda una trama i la dona a totes les gravacions en curs.
    def _push_row(self, row):
        self.frames.append(row)
        self.frame_count += 1
        for lst in self.listeners:
            lst.append(row)

    # Text d'un signe (de signs.json); el repòs no té text.
    def text_of(self, key):
        if key == self.rest_key:
            return ""
        return self.signs.get(key, {}).get("text", key)

    # Comunicació amb la pàgina web
    # Envia un missatge a totes les pàgines obertes.
    def broadcast(self, msg):
        if not self.clients:
            return
        data = json.dumps(msg)
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, data))

    async def _send(self, ws, data):
        try:
            await ws.send(data)
        except Exception:
            self.clients.discard(ws)

    # Estat general que es mostra a la capçalera de la pàgina (guants, calibratge, model, ajustos...).
    def state(self):
        rs = self.rstatus or {}
        return {"type": "state", "left": bool(self.boards["left"]), "right": bool(self.boards["right"]),
                "left_port": getattr(self.boards["left"], "port", ""), "right_port": getattr(self.boards["right"], "port", ""),
                "fps": round(self.fps, 1), "left_age": self._left_age(), "cal": bool(self.cal), "cal_sent": self.cal_sent,
                "bias_ok": {k: (v or {}).get("bias_ok", 1) for k, v in self.hello.items()},
                "session": self.session, "person": self.person, "activity": self.activity,
                "model_labels": self.labels, "thresholds": self.thresholds_now, "model_ok": self.rest_key in self.labels,
                "infer_ms": rs.get("infer_ms"), "bad_lines": rs.get("bad_lines"),
                "counts": self.counts, "signs": self.signs_data["signs"], "rest_key": self.rest_key,
                "target": self.signs_data.get("takes_target", 40), "demo": bool(self.demo),
                "data_dir": self.data_dir, "page_version": PAGE_VERSION}

    # Antiguitat (ms) de les últimes dades del guant esquerre.
    def _left_age(self):
        if not self.frames:
            return None
        return int(self.frames[-1][-1])

    # Cada 100 ms envia a la pàgina la flexió dels dits (panell "Sensors"); cada segon, l'estat general.
    async def ui_loop(self):
        last_n, last_t = 0, time.time()
        k = 0
        while True:
            await asyncio.sleep(0.1)
            k += 1
            if self.frames and self.cal:
                f = C.features(np.array([self.frames[-1]]), self.cal)[0]
                self.broadcast({"type": "live", "r": [round(x, 3) for x in f[:5]], "l": [round(x, 3) for x in f[17:22]],
                                "rz": round(float(f[10]), 2), "lz": round(float(f[27]), 2)})
            elif self.frames:
                row = self.frames[-1]
                self.broadcast({"type": "live_raw", "r": [row[COL["r_f%d" % i]] for i in range(5)],
                                "l": [row[COL["l_f%d" % i]] for i in range(5)]})
            if k % 10 == 0:
                now = time.time()
                self.fps = (self.frame_count - last_n) / (now - last_t)
                last_n, last_t = self.frame_count, now
                self.broadcast(self.state())

    # Una pàgina s'ha connectat, se li envia l'estat i s'escolten les seves ordres.
    async def ws_handler(self, ws, path=None):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps(self.state()))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                await self.command(msg)
        except Exception:
            pass
        finally:
            self.clients.discard(ws)

    # Ordres que arriben de la pàgina (botons i tecles).
    async def command(self, m):
        c = m.get("cmd")
        if c == "stop":
            self.stop_flag = True
        elif c == "discard":
            self.discard_flag = True
        elif c == "thresholds":
            # Nous ajustos de decisió (pestanya Provar) que s'envien al guant dret.
            try:
                th = dict(DECISION_DEFAULT, **(self.thresholds or {}))
                th["t_on"] = float(m.get("t_on", th["t_on"]))
                th["t_rel"] = float(m.get("t_rel", th["t_rel"]))
                th["hold"] = int(m.get("hold", th["hold"]))
                th["delay"] = int(m.get("delay", th["delay"]))
            except (ValueError, TypeError):
                return
            self.thresholds = th
            self.send_thresholds()
        elif c == "undo":                                       # Traduir, la tecla de retrocés esborra l'últim signe
            if self.activity == "translate" and not self.frase.empty():
                self.frase.undo()
                self.send_sentence(undo=True)
        elif c == "person":
            self.person = "".join(ch for ch in m.get("name", "") if ch.isalnum())[:20] or "jo"
        elif c in ("calibrate", "zero", "record", "rest", "test", "translate"):
            # Activitats llargues només n'hi pot haver una alhora.
            if self.task and not self.task.done():
                self.broadcast({"type": "error", "msg": "Ja hi ha una activitat en marxa (prem Atura)."})
                return
            self.stop_flag = self.discard_flag = False
            fn = {"calibrate": self.calibrate, "zero": self.quick_zero, "record": self.record_block,
                  "rest": self.record_rest, "test": self.run_test, "translate": self.translate}[c]
            self.task = asyncio.ensure_future(self._guard(fn, m))

    # Executa una activitat i, passi el que passi (fins i tot un error), deixa l'aplicació en repòs.
    async def _guard(self, fn, m):
        try:
            await fn(m)
        except Exception as e:                                  # la pàgina no es pot quedar mai encallada
            import traceback
            traceback.print_exc()
            self.broadcast({"type": "error", "msg": "Error: %s" % e})
        finally:
            self.activity = "idle"
            if self.demo:
                self.demo.autoplay = False
            self.broadcast({"type": "idle"})
            self.broadcast(self.state())

    # Funcions auxiliars
    # Comprova que hi ha guant dret i que arriben dades de l'esquerre; si no, torna el missatge d'error.
    def need_gloves(self):
        if not self.boards["right"]:
            return "El guant dret no està connectat."
        if self._left_age() is None or self._left_age() > C.LEFT_MAX_AGE_MS:
            return "No arriben dades del guant esquerre."
        return None

    # Grava totes les trames que arriben durant 'sec' segons.
    async def capture(self, sec):
        rows = []
        self.listeners.append(rows)
        try:
            await asyncio.sleep(sec)
        finally:
            self.listeners.remove(rows)
        return np.array(rows) if rows else np.zeros((0, len(C.RAW_COLUMNS)))

    # Mostra un missatge amb compte enrere de 'sec' segons; s'interromp si es prem Atura.
    async def countdown(self, msg, sec):
        self.broadcast(dict(msg, secs=sec))
        t_end = time.time() + sec
        while time.time() < t_end:
            if self.stop_flag:
                raise asyncio.CancelledError()
            await asyncio.sleep(0.05)

    def send_cal(self):
        if self.boards["right"] and self.cal:
            self.boards["right"].write(C.calibration_command(self.cal) + "\n")

    # Envia al guant dret la línia T amb els ajustos de decisió.
    def send_thresholds(self):
        if self.boards["right"] and self.thresholds:
            t = self.thresholds
            self.boards["right"].write(C.seal("T,%.2f,%.2f,%d,%d" % (t["t_on"], t["t_rel"], t["hold"], t["delay"])) + "\n")

    # la frase a la pantalla
    def send_sentence(self, undo=False):
        """Envia la frase reconstruïda. Les paraules noves es diuen en veu alta de seguida; les que només han
        canviat (per exemple, estimo -> t'estimo) no, i en lloc d'això, al final es torna a llegir tota la frase."""
        words = self.frase.render()
        old_ids = {w["id"] for w in self.frase_words}
        new = [w for w in words if w["id"] not in old_ids]
        speak = " ".join(w["text"] for w in new) if (new and not undo) else None
        last = self.frase.tokens[-1][1] if self.frase.tokens else ""
        if not new and not undo and last.startswith("lletra_"):
            speak = last[len("lletra_"):].upper()                  # una lletra afegida a una paraula lletrejada
        if speak:
            self.spoken += [(w["id"], w["text"]) for w in new]
        if undo:                                                    # s'obliden les paraules esborrades
            ids = {w["id"] for w in words}
            self.spoken = [(i, t) for i, t in self.spoken if i in ids]
        self.frase_words = words
        self.broadcast({"type": "sentence", "words": words, "final": False, "speak": speak})

    # La frase s'ha acabat: s'hi posa el punt i, si alguna paraula ha canviat després de dir-la,
    # es torna a llegir sencera. Després es comença una frase nova.
    def end_sentence(self):
        words = self.frase.render(final=True)
        norm = lambda ws: [w.lower().strip(",.?") for w in ws]
        final_words = [w["text"] for w in words]
        speak = None
        if self.reread and norm(final_words) != norm([t for i, t in self.spoken]):
            speak = self.frase.text(final=True)          # alguna cosa ha canviat després de dir-la: es llegeix tota
        self.broadcast({"type": "sentence", "words": words, "final": True, "speak": speak,
                        "text": self.frase.text(final=True)})
        self.frase = Frase({k: v.get("text", k) for k, v in self.signs.items()})
        self.frase_words, self.spoken, self.rest_run = [], [], 0

    # Envia el calibratge al guant dret i espera (fins a 3 s) que el confirmi.
    async def push_cal(self):
        self.cal_sent = False
        self.cal_waiter = self.loop.create_future()
        self.send_cal()
        try:
            await asyncio.wait_for(self.cal_waiter, 3.0)
            return True
        except asyncio.TimeoutError:
            return False

    # Desa el calibratge de la sessió a dataset/calibrations.
    def save_cal(self):
        d = os.path.join(self.data_dir, "calibrations")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "%s.json" % self.session), "w", encoding="utf-8") as fh:
            json.dump(self.cal, fh, indent=1)

    # Compta les preses i els segons de repòs gravats de cada signe (per a les barres de la pestanya Gravar).
    def scan_counts(self):
        counts = {}
        for js in glob.glob(os.path.join(self.data_dir, "raw", "**", "*.json"), recursive=True):
            try:
                with open(js, encoding="utf-8") as fh:
                    m = json.load(fh)
            except (OSError, ValueError):
                continue
            k = m.get("label")
            if m.get("kind") == "block":
                counts[k] = counts.get(k, 0) + len(m.get("cues", []))
            elif m.get("kind") == "rest":
                counts[k] = counts.get(k, 0) + int(m.get("frames", 0) / C.FS)
        self.counts = counts

    # Calibratge
    async def zero_capture(self):
        """Posició de zero: 3 s per preparar-se i 2 s de captura. Torna {'right': {...}, 'left': {...}} o
        dona un error si s'han perdut dades, si la mà s'ha mogut més de 15 graus o si no estava plana."""
        if self.demo:
            self.demo.person.calib_pose("zero", 5.0)
        await self.countdown({"type": "cal", "kind": "zero", "stage": "prep"}, 3.0)
        self.broadcast({"type": "cal", "kind": "zero", "stage": "capture", "secs": 2.0})
        rows = await self.capture(2.0)
        if len(rows) < 80 or not C.frames_valid(rows).all():
            raise ValueError("left")
        z = C.zero_heading(rows)
        for hand in ("right", "left"):
            if z[hand]["wander_deg"] > 15:
                raise ValueError("moved:" + hand)
            if z[hand]["tilt_deg"] > 35:
                raise ValueError("tilt:" + hand)
        return z

    # Calibratge complet: posició de zero i 5 repeticions de mà oberta / puny tancat (1 s cadascuna).
    async def calibrate(self, m):
        err = self.need_gloves()
        if err:
            self.broadcast({"type": "cal_failed", "msg": err})
            return
        self.activity = "calibrate"
        try:
            z = await self.zero_capture()
            reps = {"open": [], "fist": []}
            for rep in range(1, 6):
                for kind in ("open", "fist"):
                    if self.demo:
                        self.demo.person.calib_pose(kind, 3.0)
                    await self.countdown({"type": "cal", "kind": kind, "stage": "prep", "rep": rep, "reps": 5}, 2.0)
                    self.broadcast({"type": "cal", "kind": kind, "stage": "capture", "secs": 1.0, "rep": rep, "reps": 5})
                    rows = await self.capture(1.0)
                    if len(rows) < 35 or not C.frames_valid(rows).all():
                        raise ValueError("left")
                    reps[kind].append(rows)
            # Calibratge dels dits de cada mà (lsc_core.flex_calibration descarta les repeticions dolentes).
            cal, info = {}, {}
            for p, hand in (("r_", "right"), ("l_", "left")):
                cols = [COL[p + "f%d" % i] for i in range(5)]
                try:
                    o, c, inf = C.flex_calibration([r[:, cols] for r in reps["open"]], [r[:, cols] for r in reps["fist"]])
                except ValueError as e:
                    raise ValueError(str(e) + ":" + hand)
                cal[hand] = {"open": o, "closed": c, "psi0": z[hand]["psi0"]}
                info[hand] = inf
        except asyncio.CancelledError:
            self.broadcast({"type": "cal_failed", "msg": "Calibratge aturat."})
            return
        except ValueError as e:
            self.broadcast({"type": "cal_failed", "code": str(e)})
            return
        # Calibratge correcte: comença una sessió nova (data, hora i persona), s'envia al guant i es desa.
        self.cal = cal
        self.session = time.strftime("%Y%m%d-%H%M") + "_" + (self.person or "jo")
        ok = await self.push_cal()
        self.save_cal()
        self.broadcast({"type": "cal_done", "info": info, "sent": ok, "session": self.session})

    async def quick_zero(self, m=None):
        """Només torna a mesurar la direcció "endavant" (la direcció del giroscopi es desvia una mica amb el temps)."""
        if not self.cal:
            self.broadcast({"type": "cal_failed", "msg": "Primer cal calibrar (botó Calibrar)."})
            return False
        self.activity = "zero"
        try:
            z = await self.zero_capture()
        except asyncio.CancelledError:
            return False
        except ValueError as e:
            self.broadcast({"type": "cal_failed", "code": str(e)})
            return False
        for hand in ("right", "left"):
            self.cal[hand]["psi0"] = z[hand]["psi0"]
        ok = await self.push_cal()
        self.save_cal()
        self.broadcast({"type": "zero_done", "sent": ok})
        return True

    # Enregistrament
    # Grava un bloc de preses d'un signe: repòs -> "ARA!" -> signe -> comprovació de la presa. Les preses
    # dolentes es repeteixen fins a tenir-ne les que s'han demanat. Al final es desa tot el bloc.
    async def record_block(self, m):
        key = m.get("sign")
        if key not in self.signs:
            return
        err = self.need_gloves() or (None if self.cal else "Primer cal calibrar.")
        if err:
            self.broadcast({"type": "error", "msg": err})
            return
        n_takes = int(m.get("takes", 10))
        sign = self.signs[key]
        perform = PERFORM_S[sign["type"]]
        if m.get("rezero", True) and not await self.quick_zero():   # zero ràpid abans de cada bloc
            return
        self.activity = "record"
        rows = []
        cues, bad, results = [], [], []         # senyals de les preses bones, preses dolentes i resultats
        self.listeners.append(rows)
        base = {"type": "rec", "sign": key, "text": sign["text"], "how": sign["how"], "kind": sign["type"],
                "n": n_takes}
        last = None
        try:
            rest = REST_S + (2.0 if m.get("rezero", True) else 0.0)   # després del zero amb les mans planes cal temps per tornar a repòs
            while len(cues) < n_takes:
                await self.countdown(dict(base, stage="rest", done=len(cues), last=last), rest)
                cue = len(rows)                                     # trama del senyal "ARA!"
                if self.demo:
                    self.demo.perform(key, delay=random.uniform(.2, .6), hold=perform - 1.2)
                await self.countdown(dict(base, stage="go", done=len(cues)), perform)
                self.broadcast(dict(base, stage="rest", done=len(cues), secs=CHECK_S + NEXT_S, last=None))
                await asyncio.sleep(CHECK_S)
                # Comprovació de la presa (la mateixa funció que farà servir build_dataset.py).
                a = max(0, cue - int(PRE_S * C.FS))
                seg = np.array(rows[a:])
                if not C.frames_valid(seg).all():
                    res = {"ok": False, "reason": "s'han perdut dades del guant (esquerre?)"}
                else:
                    res = C.analyse_take(C.features(seg, self.cal), cue - a, sign)
                if self.discard_flag and cues:                     # tecla de retrocés: "descarta la presa anterior"
                    bad.append({"cue": cues.pop(), "reason": "descartat"})
                    results.pop()
                self.discard_flag = False
                if res["ok"]:
                    cues.append(cue)
                    results.append({k: (v + a if isinstance(v, int) and k in ("start", "end", "hold_start", "hold_end") else v)
                                    for k, v in res.items() if k in ("start", "end", "hold_start", "hold_end", "reason")})
                else:
                    bad.append({"cue": cue, "reason": res["reason"]})
                last = {"ok": res["ok"], "reason": res["reason"]}
                rest = NEXT_S
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            self.listeners.remove(rows)
        # Desa el bloc (CSV + JSON amb el calibratge, els senyals i les preses rebutjades).
        if cues:
            folder = os.path.join(self.data_dir, "raw", self.session)
            name = C.save_take(folder, key, self.session, rows,
                               {"kind": "block", "cues": cues, "bad": bad, "results": results,
                                "calibration": self.cal, "perform_s": perform, "rest_s": REST_S,
                                "sign_type": sign["type"], "hands": sign.get("hands", 1)})
            self.scan_counts()
            self.broadcast({"type": "saved", "file": name, "takes": len(cues), "sign": key})
        else:
            self.broadcast({"type": "saved", "file": None, "takes": 0, "sign": key})

    # Grava repòs durant 'seconds' segons. Cada 6 s la pantalla demana una activitat diferent sense signar
    # (rest_prompts de signs.json), perquè el repòs sigui variat.
    async def record_rest(self, m):
        err = self.need_gloves() or (None if self.cal else "Primer cal calibrar.")
        if err:
            self.broadcast({"type": "error", "msg": err})
            return
        secs = float(m.get("seconds", 60))
        if m.get("rezero", True) and not await self.quick_zero():
            return
        prompts = list(self.signs_data.get("rest_prompts", ["Repòs"]))
        random.shuffle(prompts)
        self.activity = "rest"
        rows = []
        marks = []
        self.listeners.append(rows)
        t0 = time.time()
        try:
            i = 0
            while time.time() - t0 < secs:
                p = prompts[i % len(prompts)]
                marks.append({"frame": len(rows), "prompt": p})
                if self.demo:
                    self.demo.person.random_rest_activity(6.0)
                await self.countdown({"type": "rest", "prompt": p, "left": round(secs - (time.time() - t0))},
                                     min(6.0, secs - (time.time() - t0)))
                i += 1
        except asyncio.CancelledError:
            pass
        finally:
            self.listeners.remove(rows)
        if len(rows) > C.WINDOW * 2:                               # només si hi ha almenys dues finestres
            folder = os.path.join(self.data_dir, "raw", self.session)
            name = C.save_take(folder, self.rest_key, self.session, rows,
                               {"kind": "rest", "prompts": marks, "calibration": self.cal})
            self.scan_counts()
            self.broadcast({"type": "saved", "file": name, "seconds": round(len(rows) / C.FS), "sign": self.rest_key})

    # Prova i traducció
    # Prova en temps real: n signes a l'atzar, cadascun després d'una pausa de repòs aleatòria. Per a cada
    # signe es mira què diu el guant en els 4,5 s següents. Al final es desa el resultat a dataset/tests.
    async def run_test(self, m):
        err = self.need_gloves() or (None if self.cal else "Primer cal calibrar.") or \
            (None if self.rest_key in self.labels else "El guant dret no té cap model amb la classe 'repos'.")
        if err:
            self.broadcast({"type": "error", "msg": err})
            return
        keys = [k for k in m.get("signs") or list(self.signs) if k in self.labels]
        n = int(m.get("n", 20))
        self.activity = "test"
        results = []
        false_alarms = []
        try:
            for i in range(n):
                gap = random.uniform(1.5, 5.0)                 # moment a l'atzar: qualsevol punt de les finestres de 200 ms
                t_rest = time.time()
                await self.countdown({"type": "test", "stage": "rest", "i": i, "n": n}, gap)
                false_alarms += [k for t, k in self.emits if t >= t_rest]    # signes dits durant el repòs
                key = random.choice(keys)
                t0 = time.time()
                if self.demo:
                    self.demo.perform(key, delay=random.uniform(.3, .8))
                await self.countdown({"type": "test", "stage": "go", "i": i, "n": n, "sign": key,
                                      "text": self.signs[key]["text"], "kind": self.signs[key]["type"]}, 4.5)
                got = [(t, k) for t, k in self.emits if t >= t0]
                said = got[0][1] if got else None                  # el primer signe dit
                lat = round(got[0][0] - t0, 2) if got else None    # retard des que la pantalla l'ha demanat
                extra = [k for t, k in got[1:]]                    # signes de més
                results.append({"want": key, "got": said, "latency": lat, "extra": extra})
                self.broadcast({"type": "test", "stage": "result", "i": i, "n": n, "want": key, "got": said,
                                "latency": lat, "extra": extra,
                                "want_text": self.signs[key]["text"], "got_text": self.text_of(said) if said else "—"})
        except asyncio.CancelledError:
            pass
        # Resum: encerts, no detectats, errors, signes de més, falses alarmes i retard mitjà.
        ok = sum(r["want"] == r["got"] for r in results)
        summary = {"type": "test_done", "n": len(results), "correct": ok,
                   "missed": sum(r["got"] is None for r in results),
                   "wrong": [[self.signs[r["want"]]["text"], self.text_of(r["got"])] for r in results if r["got"] and r["got"] != r["want"]],
                   "double": sum(len(r["extra"]) for r in results), "false_alarms": len(false_alarms),
                   "latency": round(float(np.mean([r["latency"] for r in results if r["latency"] is not None])), 2)
                   if any(r["latency"] is not None for r in results) else None}
        os.makedirs(os.path.join(self.data_dir, "tests"), exist_ok=True)
        with open(os.path.join(self.data_dir, "tests", time.strftime("%Y%m%d-%H%M%S") + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results, "false_alarms": false_alarms}, fh, indent=1)
        self.broadcast(summary)

    # Pantalla de traducció: comença una frase buida i espera fins que es prem Atura. Les paraules
    # s'afegeixen a _on_line_loop cada vegada que el guant diu un signe.
    async def translate(self, m):
        try:
            self.sentence_end_windows = max(3, int(round(float(m.get("end_s", SENTENCE_END_S)) / 0.2)))
        except (TypeError, ValueError):
            self.sentence_end_windows = int(round(SENTENCE_END_S / 0.2))
        self.reread = bool(m.get("reread", True))
        self.frase = Frase({k: v.get("text", k) for k, v in self.signs.items()})
        self.frase_words, self.spoken, self.rest_run = [], [], 0
        self.activity = "translate"
        if self.demo:
            self.demo.autoplay = True
        try:
            while not self.stop_flag:
                await asyncio.sleep(0.2)
        finally:
            if not self.frase.empty():
                self.end_sentence()


# Programa principal
# Servidor de la pàgina web que demana al navegador que no la guardi a la memòria cau
# (així sempre es carrega la versió actual).
class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")          # sempre la pàgina actual, sense haver de prémer Ctrl+F5
        super().end_headers()


# Serveix la pàgina web (lsc_app.html) al port 8000.
def serve_http(directory):
    handler = functools.partial(NoCacheHandler, directory=directory)
    handler.log_message = lambda *a, **k: None

    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    Srv(("", HTTP_PORT), handler).serve_forever()


# Engega el Hub, els guants (reals o simulats), el bucle de la pàgina i el servidor WebSocket, i obre el navegador.
async def main_async(args):
    import websockets
    hub = Hub(args)
    hub.loop = asyncio.get_running_loop()
    hub.scan_counts()
    if args.demo:
        hub.demo = DemoGloves(hub)
        asyncio.ensure_future(hub.demo.run())
    else:
        threading.Thread(target=port_scanner, args=(hub,), daemon=True).start()
    asyncio.ensure_future(hub.ui_loop())
    async with websockets.serve(hub.ws_handler, "0.0.0.0", WS_PORT, max_size=2 ** 22):
        print("LSC app running: http://localhost:%d   (data folder: %s)" % (HTTP_PORT, hub.data_dir))
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open("http://localhost:%d/lsc_app.html" % HTTP_PORT)).start()
        await asyncio.Future()                              # funciona fins que es tanca el programa


# Opcions de la línia d'ordres.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="simulated gloves (no hardware)")        # guants simulats
    ap.add_argument("--data", default=os.path.join(HERE, "dataset"), help="where recordings are saved")   # carpeta de les gravacions
    ap.add_argument("--person", default="jo", help="who wears the gloves (goes into the session name)")  # qui porta els guants
    ap.add_argument("--no-browser", action="store_true")                                        # no obre el navegador
    args = ap.parse_args()
    threading.Thread(target=serve_http, args=(HERE,), daemon=True).start()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
