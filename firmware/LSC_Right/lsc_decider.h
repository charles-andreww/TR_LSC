//Lògica de decisió, decideix quan s'ha de "dir" un signe, i només una vegada.
//
// El model analitza els darrers 2,0 s cada 200 ms, de manera que UN mateix signe apareix
// en diverses finestres seguides. Les finestres que només contenen una part d'un signe
// s'han entrenat com a "repòs" (build_dataset.py). Per tant, mentre la persona comença
// o acaba un signe, el model diu "repòs", i quan la finestra conté el signe sencer, diu el signe.
//
// Aquest fitxer:
//   * Un signe passa a ser CANDIDAT quan és la classe guanyadora amb una probabilitat
//     d'almenys tOn en 2 de les 3 darreres finestres i no està bloquejat.
//     (Una sola finestra encertada per casualitat no n'hi ha prou.)
//   * El candidat es diu `delay` finestres més tard. Si durant aquesta espera un altre signe
//     té més suport, es diu aquest altre. (Evita que un signe equivocat aparegui just abans del correcte.)
//   * Després de dir un signe, durant `hold` finestres no se'n pot dir cap. (Evita que el final
//     d'un signe i el retorn a repòs semblin un altre signe, per exemple BON DIA -> QUAN.)
//   * L'últim signe dit queda "bloquejat" fins que el model diu "repòs" (amb probabilitat
//     d'almenys tRel) o fins que es diu un altre signe. Per tant:
//       - un signe que el model veu en 5 finestres seguides es diu una sola vegada;
//       - per dir el mateix signe dues vegades cal fer una petita pausa entre mig.
// Els valors per defecte (tOn 0,70, hold 5 = 1,0 s, delay 2 = 0,4 s) es van triar reproduint
// totes les gravacions amb el model real (tests/replay_decider.py). Es poden canviar des de l'aplicació.
// C pur, sense biblioteques d'Arduino (les proves de l'ordinador compilen aquest mateix fitxer).
#pragma once

#define LSC_DEC_MAX_CLASSES 32    // nombre màxim de classes que admet
#define LSC_DEC_HIST 3            // quantes finestres recorda (les 3 darreres)
#define LSC_DEC_HOLD_DEFAULT 5    // finestres de silenci després d'un signe (5 x 200 ms = 1,0 s)
#define LSC_DEC_DELAY_DEFAULT 2   // finestres que espera un candidat (2 x 200 ms = 0,4 s)

// Estat de la lògica de decisió.
typedef struct {
  int n, rest;            // nombre de classes, i posició de la classe "repòs"
  float tOn, tRel;        // llindar per ser candidat, i llindar de "repòs" per desbloquejar
  int hold, delay;        // finestres de silenci després d'un signe, i finestres d'espera d'un candidat
  int lab[LSC_DEC_HIST];  // classe guanyadora de les darreres finestres ([0] = la més recent), -1 = buit
  float p[LSC_DEC_HIST];  // probabilitat d'aquella classe guanyadora
  int locked;             // últim signe dit (bloquejat), -1 = cap
  int holdLeft;           // finestres de silenci que queden
  int pend, pendLeft;     // candidat que espera (-1 = cap) i finestres que li queden
} LscDecider;

// Esborra la memòria: oblida les finestres anteriors, el candidat i el bloqueig.
// (Els llindars i els temps es mantenen.)
static inline void lscDeciderReset(LscDecider* d) {
  for (int i = 0; i < LSC_DEC_HIST; i++) { d->lab[i] = -1; d->p[i] = 0.0f; }
  d->locked = -1;
  d->holdLeft = 0;
  d->pend = -1;
  d->pendLeft = 0;
}

// Prepara la lògica de decisió amb el nombre de classes, la posició de "repòs" i els dos llindars.
// Els temps comencen amb els valors per defecte.
static inline void lscDeciderInit(LscDecider* d, int n, int rest, float tOn, float tRel) {
  d->n = n; d->rest = rest; d->tOn = tOn; d->tRel = tRel;
  d->hold = LSC_DEC_HOLD_DEFAULT; d->delay = LSC_DEC_DELAY_DEFAULT;
  lscDeciderReset(d);
}

// Canvia els temps de silenci i d'espera (en finestres de 200 ms). Els valors negatius passen a 0.
static inline void lscDeciderSetTiming(LscDecider* d, int hold, int delay) {
  d->hold = hold < 0 ? 0 : hold;
  d->delay = delay < 0 ? 0 : delay;
}

// Suport de la classe c en les darreres finestres: suma de les seves probabilitats guanyadores
// d'almenys tOn. Si votes no és nul, hi guarda en quantes finestres ha guanyat.
static inline float lscDeciderStrength(const LscDecider* d, int c, int* votes) {
  float s = 0.0f;
  int v = 0;
  for (int i = 0; i < LSC_DEC_HIST; i++)
    if (d->lab[i] == c && d->p[i] >= d->tOn) { s += d->p[i]; v++; }
  if (votes) *votes = v;
  return s;
}

// Diu el signe c: el bloqueja i comença el silenci.
static inline int lscDeciderSay(LscDecider* d, int c) {
  d->locked = c;
  d->holdLeft = d->hold;
  return c;
}

// Rep les n probabilitats de la finestra més nova.
// Retorna la classe que s'ha de dir ara, o -1 si no s'ha de dir res.
static inline int lscDeciderStep(LscDecider* d, const float* p) {
  //  Busca la classe amb la probabilitat més alta i la guarda al començament de la memòria.
  int a = 0;
  for (int i = 1; i < d->n; i++) if (p[i] > p[a]) a = i;
  for (int i = LSC_DEC_HIST - 1; i > 0; i--) { d->lab[i] = d->lab[i - 1]; d->p[i] = d->p[i - 1]; }
  d->lab[0] = a; d->p[0] = p[a];

  //Si guanya "repòs" amb prou seguretat, es desbloqueja l'últim signe dit.
  if (a == d->rest && p[a] >= d->tRel) d->locked = -1;
  // Just després de dir un signe, silenci: no es diu res.
  if (d->holdLeft > 0) { d->holdLeft--; return -1; }

  // És candidat? Ha de guanyar amb prou probabilitat en almenys 2 de les 3 darreres finestres
  //    i no ha d'estar bloquejat.
  int votes = 0;
  int cand = -1;
  if (a != d->rest && p[a] >= d->tOn && a != d->locked) {
    lscDeciderStrength(d, a, &votes);
    if (votes >= 2) cand = a;
  }

  // Si ja hi ha un candidat esperant: si el nou té més suport, el substitueix.
  //    Quan s'acaba l'espera, es diu el candidat.
  if (d->pend >= 0) {
    if (cand >= 0 && cand != d->pend &&
        lscDeciderStrength(d, cand, 0) > lscDeciderStrength(d, d->pend, 0)) d->pend = cand;
    if (--d->pendLeft <= 0) {
      int e = d->pend;
      d->pend = -1;
      if (e == d->locked) return -1;
      return lscDeciderSay(d, e);
    }
    return -1;
  }
  // Un candidat nou comença a esperar (o es diu de seguida si no hi ha espera).
  if (cand >= 0) {
    if (d->delay <= 0) return lscDeciderSay(d, cand);
    d->pend = cand;
    d->pendLeft = d->delay;
  }
  return -1;
}
