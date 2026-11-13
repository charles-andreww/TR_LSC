"""
converteix els signes que diu el guant en una frase en català, paraula a paraula.

Els signes que reconeix el guant no porten cap marca de persona ni de temps: la persona es diu
assenyalant (JO, TU). Aquest fitxer torna a construir la frase en català després de cada signe nou,
de manera que les paraules anteriors poden canviar quan arriba més informació:

    JO                  ->  Jo
    JO ESTIMAR          ->  Jo estimo
    JO ESTIMAR TU       ->  Jo t'estimo            (TU darrere d'un verb com ESTIMAR = el complement)
    TU ANAR ON          ->  On vas tu?             (en les preguntes, el subjecte va darrere del verb)
    VOLER AIGUA         ->  Vull aigua             (sense subjecte: "jo" en les afirmacions, "tu" en les preguntes)
    JO NO VOLER         ->  Jo no vull
    JO VOLER ANAR       ->  Jo vull anar           (només es conjuga el primer verb)
    HOLA JO VOLER AIGUA SI_US_PLAU  ->  Hola, jo vull aigua, si us plau.

Cada paraula de la frase guarda l'identificador del signe d'on surt, perquè la pantalla pugui animar
exactament què ha aparegut, què ha canviat i què ha desaparegut (per exemple, "tu" que es fon dins de "t'estimo").
Python sol, sense biblioteques. Si s'executa directament, mostra els exemples.
"""

# Vocabulari
PRONOUNS = {"jo": 1, "tu": 2}                           # pronoms i la persona que indiquen (1a o 2a)
PRONOUN_TEXT = {1: "jo", 2: "tu"}
CLITIC = {1: ("em ", "m'"), 2: ("et ", "t'")}           # pronoms febles: davant de consonant / davant de vocal

# Present d'indicatiu de cada verb: infinitiu, jo, tu, (ell/ella)
VERBS = {
    "anar":    ("anar", "vaig", "vas", "va"),
    "fer":     ("fer", "faig", "fas", "fa"),
    "voler":   ("voler", "vull", "vols", "vol"),
    "estimar": ("estimar", "estimo", "estimes", "estima"),
    "ajudar":  ("ajudar", "ajudo", "ajudes", "ajuda"),
    "necessitar": ("necessitar", "necessito", "necessites", "necessita"),
}
# Verbs que normalment tenen una persona com a complement: un pronom fet DARRERE d'aquests verbs
# és el complement (ESTIMAR TU = t'estimo).
PERSON_OBJECT = {"estimar", "ajudar"}

WH = {"que": "què", "on": "on", "quan": "quan", "qui": "qui"}       # paraules interrogatives
NEGATION = {"no"}
COURTESY = {"hola": "hola", "bon_dia": "bon dia", "gracies": "gràcies", "si_us_plau": "si us plau"}   # salutacions i cortesia
OTHER = {"si": "sí", "aigua": "aigua", "ajuda": "ajuda"}            # altres paraules amb un text especial

# Persona que s'utilitza quan un verb no té subjecte: les afirmacions parlen de "jo" i les preguntes, de "tu".
DEFAULT_PERSON_STATEMENT = 1
DEFAULT_PERSON_QUESTION = 2

VOWELS = set("aeiouàèéíòóúAEIOUÀÈÉÍÒÓÚ")


# Comprova si una paraula comença per vocal (una h inicial no compta: "h" + vocal = vocal).
def _starts_with_vowel(word):
    w = word.lower()
    if w.startswith("h") and len(w) > 1:
        w = w[1:]
    return bool(w) and w[0] in VOWELS


# Si el signe és una lletra de l'alfabet dactilològic ("lletra_A"), torna la lletra ("A"); si no, None.
def _letter(key):
    return key[len("lletra_"):].upper() if key.startswith("lletra_") else None


class Frase:
    """Una frase en construcció. Es crida add(signe) després de cada signe i render() dona les paraules que s'han de mostrar."""

    def __init__(self, texts=None):
        self.tokens = []            # signes rebuts, en ordre: [(identificador, signe)]
        self.next_id = 0
        self.texts = texts or {}    # signe -> text de signs.json (per als signes que aquest fitxer no coneix)

    # Afegeix un signe nou a la frase, amb un identificador propi.
    def add(self, key):
        self.tokens.append(("w%d" % self.next_id, key))
        self.next_id += 1

    # Esborra l'últim signe (tecla de retrocés).
    def undo(self):
        if self.tokens:
            self.tokens.pop()

    def empty(self):
        return not self.tokens

    # Les regles
    def render(self, final=False):
        """Torna una llista de {"id", "text", "punct"} (identificador, text i signe de puntuació) en l'ordre de lectura."""
        # Classifica cada signe. Les lletres fetes seguides formen una sola paraula lletrejada.
        items = []
        for tid, key in self.tokens:
            L = _letter(key)
            if L and items and items[-1]["kind"] == "spell":
                items[-1]["text"] += L                            # s'afegeix a la paraula lletrejada anterior
                continue
            if L:
                items.append({"id": tid, "kind": "spell", "key": key, "text": L})
            elif key in PRONOUNS:
                items.append({"id": tid, "kind": "pron", "key": key, "person": PRONOUNS[key]})
            elif key in VERBS:
                items.append({"id": tid, "kind": "verb", "key": key})
            elif key in WH:
                items.append({"id": tid, "kind": "wh", "key": key, "text": WH[key]})
            elif key in NEGATION:
                items.append({"id": tid, "kind": "neg", "key": key, "text": "no"})
            elif key in COURTESY:
                items.append({"id": tid, "kind": "court", "key": key, "text": COURTESY[key]})
            else:
                items.append({"id": tid, "kind": "word", "key": key,
                              "text": OTHER.get(key, self.texts.get(key, key).lower())})

        #Les paraules de cortesia fetes abans de qualsevol altra paraula obren la frase ("lead");
        #    les que vénen després la tanquen ("tail"). La resta és el nucli de la frase ("core").
        first_core = next((i for i, it in enumerate(items) if it["kind"] != "court"), len(items))
        lead = [it for it in items[:first_core]]
        core = [it for it in items[first_core:] if it["kind"] != "court"]
        tail = [it for it in items[first_core:] if it["kind"] == "court"]

        verbs = [it for it in core if it["kind"] == "verb"]
        question = any(it["kind"] == "wh" for it in core)       # és una pregunta si hi ha QUÈ, ON, QUAN o QUI
        words = []
        if not verbs:
            # Encara no hi ha cap verb: les paraules es mostren en l'ordre en què s'han fet.
            for it in core:
                if it["kind"] == "pron":
                    words.append(self._w(it, PRONOUN_TEXT[it["person"]]))
                else:
                    words.append(self._w(it, it["text"]))
        else:
            v0 = verbs[0]                                       # el primer verb és el que es conjuga
            iv = core.index(v0)
            prons = [it for it in core if it["kind"] == "pron"]
            subj = obj = None                                   # subjecte i complement
            before = [p for p in prons if core.index(p) < iv]   # pronoms fets abans del verb
            after = [p for p in prons if core.index(p) > iv]    # pronoms fets després del verb
            # El primer pronom abans del verb és el subjecte; un segon pronom abans del verb, el complement.
            if before:
                subj = before[0]
                if len(before) > 1:
                    obj = before[1]
            # Un pronom després del verb: és el subjecte si encara no n'hi ha cap i el verb no és com ESTIMAR;
            # si no, és el complement.
            for p in after:
                if subj is None and v0["key"] not in PERSON_OBJECT:
                    subj = p
                elif obj is None:
                    obj = p
            extra_prons = [p for p in prons if p is not subj and p is not obj]
            # Persona del verb: la del subjecte o, si no n'hi ha, "jo" (afirmació) o "tu" (pregunta).
            person = subj["person"] if subj else (DEFAULT_PERSON_QUESTION if question else DEFAULT_PERSON_STATEMENT)

            verb_text = VERBS[v0["key"]][person] if person in (1, 2, 3) else VERBS[v0["key"]][0]
            if obj is not None:
                # El complement es converteix en pronom feble davant del verb: "et" / "t'", "em" / "m'".
                cons, vow = CLITIC[obj["person"]]
                verb_text = (vow if _starts_with_vowel(verb_text) else cons) + verb_text

            # Ordre en català: subjecte, paraula interrogativa, "no", (pronom feble +) verb, altres verbs, la resta.
            # En una pregunta, el subjecte va darrere del verb: "On vas tu?", "Què fas tu?"
            if subj is not None and not question:
                words.append(self._w(subj, PRONOUN_TEXT[subj["person"]]))
            for it in core:
                if it["kind"] == "wh":
                    words.append(self._w(it, it["text"]))
            for it in core:
                if it["kind"] == "neg":
                    words.append(self._w(it, "no"))
            words.append(self._w(v0, verb_text))
            for it in verbs[1:]:
                words.append(self._w(it, VERBS[it["key"]][0]))  # els altres verbs, en infinitiu
            if subj is not None and question:
                words.append(self._w(subj, PRONOUN_TEXT[subj["person"]]))
            for it in core:
                if it["kind"] in ("word", "spell"):
                    words.append(self._w(it, it["text"]))
            for p in extra_prons:
                words.append(self._w(p, PRONOUN_TEXT[p["person"]]))

        out = [self._w(it, it["text"]) for it in lead] + words + [self._w(it, it["text"]) for it in tail]

        #Puntuació i majúscula inicial.
        n_lead, n_words = len(lead), len(words)
        for i, w in enumerate(out):
            if i < n_lead and i < len(out) - 1:
                w["punct"] = ","                                  # "Hola, ..."
            if n_lead + n_words <= i and i > 0 and i == n_lead + n_words and out[i - 1]["punct"] == "":
                out[i - 1]["punct"] = ","                         # "..., si us plau"
        if out:
            if question:
                out[-1]["punct"] = "?"
            elif final:
                out[-1]["punct"] = "."                            # el punt només quan la frase s'ha acabat
            w = out[0]
            if not w["text"].isupper():
                w["text"] = w["text"][:1].upper() + w["text"][1:]
        return out

    # Crea una paraula de sortida que conserva l'identificador del signe d'on surt.
    @staticmethod
    def _w(it, text):
        return {"id": it["id"], "text": text, "punct": ""}

    # La frase com a text normal (per llegir-la en veu alta o desar-la).
    def text(self, final=False):
        return " ".join(w["text"] + w["punct"] for w in self.render(final)).replace(" ,", ",")


# Si s'executa aquest fitxer directament, mostra com evolucionen unes quantes frases d'exemple.
if __name__ == "__main__":
    examples = [
        ["jo"], ["jo", "estimar"], ["jo", "estimar", "tu"], ["jo", "tu", "estimar"],
        ["tu", "ajudar", "jo"], ["estimar", "tu"], ["tu", "anar", "on"], ["on", "tu", "anar"],
        ["anar", "on"], ["que", "fer"], ["tu", "que", "fer"], ["voler", "aigua"],
        ["jo", "no", "voler"], ["jo", "voler", "anar"], ["jo", "voler", "ajudar", "tu"],
        ["hola"], ["hola", "jo", "voler", "aigua", "si_us_plau"], ["bon_dia", "tu", "anar", "quan"],
        ["gracies"], ["si"], ["no"], ["qui"], ["tu", "estimar", "jo"], ["jo", "estimar", "jo"],
        ["lletra_C", "lletra_A", "lletra_L"], ["jo", "lletra_A", "lletra_L"], ["jo", "fer", "lletra_S"],
    ]
    for ex in examples:
        f = Frase()
        steps = []
        for k in ex:
            f.add(k)
            steps.append(f.text())
        print("%-45s %s   ->   %s" % (" ".join(k.upper() for k in ex), " | ".join(steps[:-1]), f.text(final=True)))
