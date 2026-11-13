// Què fa aquest programa:
//   * Llegeix els seus propis sensors 100 vegades per segon (glove_hw.h).
//   * Rep les línies del guant esquerre (L,...), que l'aplicació de l'ordinador li
//     reenvia pel mateix cable USB.
//   * Cada 20 ms (50 vegades per segon) ajunta la seva trama amb la trama més recent
//     del guant esquerre i:
//       - l'envia a l'ordinador com una línia F. Aquesta és la línia que es GRAVA per
//         entrenar el model, de manera que les dades d'entrenament són exactament les
//         mateixes que veu aquesta placa en temps real;
//       - la converteix en les 34 dades d'entrada del model (lsc_features.h, amb el
//         mateix càlcul que pc/lsc_core.py) i la guarda en una memòria circular de
//         2,0 s (100 trames).
//   * Cada 200 ms executa el model d'Edge Impulse sobre els darrers 2,0 s, passa les
//     probabilitats a la lògica de decisió (lsc_decider.h) i envia una línia P amb les
//     probabilitats i el signe que s'ha de dir (o -1 si no s'ha de dir res).
//
// El model només s'executa si la placa està calibrada (l'aplicació li envia el
// calibratge després de la pantalla de calibratge) i si les darreres 100 trames són
// totes vàlides (les dues IMU funcionen i les dades del guant esquerre són recents).
// Velocitat sèrie: 460800 baud. Carpeta: LSC_Right.ino, glove_hw.h, lsc_features.h, lsc_decider.h
// HAS_MODEL 0 (a sota): no cal la biblioteca d'Edge Impuls, el guant només envia trames per gravar.
// HAS_MODEL 1: cal la biblioteca d'Arduino exportada del projecte d'Edge Impulse.
// En engegar-lo: cal tenir el guant QUIET durant 2 s (calibratge del giroscopi).

// 0 = ENCARA NO HI HA MODEL: el guant només envia trames (suficient per calibrar i GRAVAR les dades).
// 1 = després d'entrenar: cal instal·lar la biblioteca d'Edge Impulse i posar-hi un 1.
#define HAS_MODEL 1

#if HAS_MODEL

#include <TR_LSC_inferencing.h>     // biblioteca del model, exportada des d'Edge Impulse
#else
// Sense model: es defineixen uns valors mínims perquè el programa compili igualment.
#define EI_CLASSIFIER_LABEL_COUNT 1
#define EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE 1
static const char* ei_classifier_inferencing_categories[] = {"none"};
#endif

// Pins de l'ESP32 on hi ha connectat cada dit: polze, índex, cor, anular, petit
// (els mateixos que al programa de prova del guant dret).
static const int GLOVE_PINS[5] = {33, 32, 35, 34, 39};

#include "glove_hw.h"          // lectura dels sensors (comú als dos guants)
#include "lsc_features.h"      // càlcul de les 34 dades d'entrada del model
#include "lsc_decider.h"       // lògica de decisió

#define BAUD             460800    // velocitat de la comunicació USB
#define WINDOW_FRAMES    100       // mida de la finestra: 100 trames = 2,0 s
#define STRIDE_FRAMES    10        // el model s'executa cada 10 trames = 200 ms
#define LEFT_MAX_AGE_MS  250       // si les dades del guant esquerre tenen més de 250 ms, es considera perdut
#define T_ON             0.70f     // probabilitat necessària (en 2 de les 3 darreres finestres) per ser candidat
#define T_REL            0.50f     // probabilitat de "repòs" que permet tornar a dir el mateix signe
#define DEC_HOLD         5         // finestres (x 200 ms) de silenci després de dir un signe
#define DEC_DELAY        2         // finestres (x 200 ms) que espera un candidat, per si n'arriba un de més fort
                                   // (tots quatre es poden canviar des de l'aplicació, pestanya Provar,
                                   //  sense tornar a programar el guant; valors triats reproduint totes
                                   //  les gravacions, vegeu tests/replay_decider.py)

#if HAS_MODEL
// Comprovacions en compilar: si el model exportat no té la forma esperada, el programa no compila.
static_assert(EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME == LSC_N_FEAT,
              "The Edge Impulse model must have 34 axes (build_dataset.py output)");      // 34 dades per trama
static_assert(EI_CLASSIFIER_RAW_SAMPLE_COUNT == WINDOW_FRAMES,
              "The Edge Impulse window must be 2000 ms at 50 Hz = 100 frames");           // finestra de 100 trames
static_assert(EI_CLASSIFIER_LABEL_COUNT <= LSC_DEC_MAX_CLASSES, "too many classes");     // màxim 32 classes
#endif

//  dades compartides
// Aquest programa té dues tasques que funcionen alhora (sensors i model) i el bucle
// principal. Les variables que comparteixen es protegeixen perquè dues parts del
// programa no les modifiquin a la vegada.
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;   // protecció per a les dades del guant esquerre
static GloveRaw leftRaw = {};                 // última trama rebuda del guant esquerre
static volatile uint32_t leftRxMs = 0;        // moment en què s'ha rebut (ms)
static volatile bool leftEver = false;        // si s'ha rebut mai alguna trama del guant esquerre
static uint32_t leftSeq = 0, leftLost = 0, badLines = 0;   // últim número de trama, trames perdudes, línies errònies
static bool leftSeqInit = false;              // si ja tenim un primer número de trama

static LscCalHand cal[2];                     // calibratge de cada mà: 0 = dreta, 1 = esquerra
static volatile bool calOk = false;           // si ja s'ha rebut el calibratge

static float ring[WINDOW_FRAMES][LSC_N_FEAT]; // memòria circular: les darreres 100 trames de 34 dades
static int ringHead = 0;                      // fila on s'escriurà la trama següent
static volatile int validRun = 0;             // quantes trames vàlides seguides hi ha a la memòria
static volatile uint32_t frameNo = 0;         // número de trama del guant dret
static SemaphoreHandle_t ringMtx;             // protecció de la memòria circular

// Cua de sortida, les tasques no escriuen directament al port USB, sinó que deixen les
// línies en aquesta cua i el bucle principal les envia. Així no s'interrompen entre elles.
#define TX_LEN 512                            // longitud màxima d'una línia (la presentació porta els noms
                                              // de totes les classes: amb 24 classes en calen uns 340)
struct TxLine { char s[TX_LEN]; };
static QueueHandle_t txQ;
static volatile uint32_t txDrop = 0;          // línies perdudes perquè la cua era plena

#if HAS_MODEL
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];   // finestra de 2 s ordenada, tal com la rep el model
#endif
static LscDecider decider;                    // estat de la lògica de decisió
static int restIndex = -1;                    // posició de la classe "repos" a la llista de classes
static volatile bool resetDecider = true;     // demana reiniciar la lògica de decisió
static volatile uint32_t inferMs = 0, inferCount = 0;   // temps de l'última execució del model (ms) i nombre d'execucions

// funcions auxiliars
// Posa una línia de text a la cua de sortida. Si la cua és plena, la línia es perd i es compta.
static void queueLine(const char* s, int n) {
  TxLine t;
  if (n <= 0 || n >= TX_LEN) return;
  memcpy(t.s, s, n); t.s[n] = 0;
  if (xQueueSend(txQ, &t, 0) != pdTRUE) txDrop++;
}

// Missatge de presentació: quin guant és, versió del programa, estat de la IMU, nombre de
// dades d'entrada, mida de la finestra, posició de "repòs", ajustos de la decisió i llista de classes del model.
static void sendHello() {
  char b[TX_LEN];
  int n = snprintf(b, sizeof(b), "{\"type\":\"hello\",\"hand\":\"right\",\"fw\":\"2.0\",\"imu\":%d,\"bias_ok\":%d,"
                   "\"axes\":%d,\"window\":%d,\"rest_index\":%d,\"t_on\":%.2f,\"t_rel\":%.2f,\"hold\":%d,\"delay\":%d,\"labels\":[",
                   hwImuOk ? 1 : 0, hwBiasOk ? 1 : 0, (int)LSC_N_FEAT,
                   (int)WINDOW_FRAMES, restIndex, decider.tOn, decider.tRel, decider.hold, decider.delay);
  for (int i = 0; HAS_MODEL && i < EI_CLASSIFIER_LABEL_COUNT && n < TX_LEN - 30; i++)   // noms de les classes
    n += snprintf(b + n, sizeof(b) - n, "%s\"%s\"", i ? "," : "", ei_classifier_inferencing_categories[i]);
  n += snprintf(b + n, sizeof(b) - n, "]}\n");
  queueLine(b, n);
}

// ------------------------------------------------------------------ tasca dels sensors (100 Hz)
// S'executa al nucli 1, cada 10 ms, sense que el model la pugui endarrerir.
static void sensorTask(void*) {
  TickType_t last = xTaskGetTickCount();
  uint8_t tick = 0;
  for (;;) {                                   // bucle infinit
    hwTick();                                  // llegeix els sensors propis
    if (++tick >= 2) {                         // cada 20 ms: una trama
      tick = 0;
      GloveRaw own, left;
      hwTakeFrame(&own);                       // trama del guant dret
      // Copia la trama més recent del guant esquerre (protegida, perquè el bucle principal la pot estar canviant).
      portENTER_CRITICAL(&mux);
      left = leftRaw;
      uint32_t rx = leftRxMs;
      bool ever = leftEver;
      portEXIT_CRITICAL(&mux);
      // Antiguitat de les dades del guant esquerre (ms). Si són massa antigues, el guant esquerre s'ha perdut.
      uint32_t age = ever ? (millis() - rx) : 65535;
      if (age > 65535) age = 65535;
      bool leftOk = ever && age <= LEFT_MAX_AGE_MS;
      if (leftOk) own.flags |= FLAG_LEFT_OK;

      // Línia F = la trama exactament com es fa servir a continuació (és el que grava l'ordinador).
      // Format: "F,número," + 16 valors de la mà dreta + "," + 16 valors de l'esquerra + ",antiguitat" + "*HH"
      char b[TX_LEN];
      int n = snprintf(b, sizeof(b), "F,%lu,", (unsigned long)frameNo);
      n += hwFormatHand(b + n, sizeof(b) - n, &own);
      b[n++] = ',';
      n += hwFormatHand(b + n, sizeof(b) - n, &left);
      n += snprintf(b + n, sizeof(b) - n, ",%lu", (unsigned long)age);
      n = hwSeal(b, n, sizeof(b));
      queueLine(b, n);

      // Dades d'entrada del model. La trama només és vàlida si hi ha calibratge, el guant
      // esquerre arriba a temps i les dues IMU funcionen.
      bool valid = calOk && leftOk && (own.flags & FLAG_IMU) && (left.flags & FLAG_IMU);
      if (valid) {
        // Els números es tornen a llegir de la línia de text que s'acaba d'enviar. Així el model
        // fa servir exactament els valors gravats (arrodonits als mateixos decimals).
        LscRawHand r, l;
        GloveRaw pr, pl;
        const char* p = b + 2;                 // salta "F,"
        while (*p && *p != ',') p++;           // salta el número de trama
        p++;
        int u = hwParseHand(p, &pr);           // 16 valors de la mà dreta
        int u2 = (u > 0) ? hwParseHand(p + u + 1, &pl) : -1;   // 16 valors de la mà esquerra
        if (u2 > 0) {
          // Copia les dades al format que fa servir lsc_features.h.
          memcpy(r.f, pr.f, sizeof r.f); memcpy(r.q, pr.q, sizeof r.q); memcpy(r.g, pr.g, sizeof r.g); memcpy(r.a, pr.a, sizeof r.a);
          memcpy(l.f, pl.f, sizeof l.f); memcpy(l.q, pl.q, sizeof l.q); memcpy(l.g, pl.g, sizeof l.g); memcpy(l.a, pl.a, sizeof l.a);
          // Calcula les 34 dades: 17 de la mà dreta i, a continuació, 17 de l'esquerra.
          float row[LSC_N_FEAT];
          lscHandFeatures(&r, &cal[0], row);
          lscHandFeatures(&l, &cal[1], row + LSC_FEAT_PER_HAND);
          // Guarda la fila a la memòria circular (substitueix la trama més antiga).
          xSemaphoreTake(ringMtx, portMAX_DELAY);
          memcpy(ring[ringHead], row, sizeof(row));
          ringHead = (ringHead + 1) % WINDOW_FRAMES;          // en arribar a 100 torna a 0
          if (validRun < 1000000) validRun++;
          xSemaphoreGive(ringMtx);
        } else {
          validRun = 0;                        // error en llegir la línia: es torna a omplir la finestra
        }
      } else {
        validRun = 0;                          // trama no vàlida: es torna a omplir la finestra
      }
      frameNo++;
    }
    vTaskDelayUntil(&last, pdMS_TO_TICKS(HW_TICK_MS));   // espera fins a completar els 10 ms exactes
  }
}

// tasca del model (cada 200 ms)
// S'executa al nucli 0, de manera que el càlcul del model no retarda la lectura dels sensors.
#if HAS_MODEL
static void inferTask(void*) {
  uint32_t lastFrame = 0;
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(5));              // comprova cada 5 ms si ja toca
    uint32_t f = frameNo;
    if (f - lastFrame < STRIDE_FRAMES) continue;   // encara no han passat 10 trames (200 ms)
    lastFrame = f;
    if (resetDecider) { lscDeciderReset(&decider); resetDecider = false; }
    // No s'executa el model si no hi ha calibratge, si el model no té classe "repos"
    // o si encara no hi ha 100 trames vàlides seguides (2 s complets).
    if (!calOk || restIndex < 0 || validRun < WINDOW_FRAMES) { lscDeciderReset(&decider); continue; }

    // Copia la memòria circular a 'features' en ordre cronològic (de la trama més antiga a la més nova).
    xSemaphoreTake(ringMtx, portMAX_DELAY);
    for (int i = 0; i < WINDOW_FRAMES; i++)                       // primer la trama més antiga
      memcpy(&features[i * LSC_N_FEAT], ring[(ringHead + i) % WINDOW_FRAMES], sizeof(float) * LSC_N_FEAT);
    xSemaphoreGive(ringMtx);

    // Executa el model d'Edge Impulse i mesura quant tarda.
    signal_t signal;
    numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    ei_impulse_result_t result = {0};
    uint32_t t0 = millis();
    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) continue;
    inferMs = millis() - t0;
    inferCount++;

    // Probabilitat de cada classe -> lògica de decisió -> signe que s'ha de dir (o -1).
    float p[EI_CLASSIFIER_LABEL_COUNT];
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) p[i] = result.classification[i].value;
    int emit = lscDeciderStep(&decider, p);

    // Línia P: "P,número de trama," + probabilitats (3 decimals) + ",signe a dir" + "*HH"
    char b[TX_LEN];
    int n = snprintf(b, sizeof(b), "P,%lu", (unsigned long)f);
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT && n < TX_LEN - 16; i++) n += snprintf(b + n, sizeof(b) - n, ",%.3f", p[i]);
    n += snprintf(b + n, sizeof(b) - n, ",%d", emit);
    n = hwSeal(b, n, sizeof(b));
    queueLine(b, n);
  }
}
#endif

// ordres de l'ordinador
// Interpreta una línia rebuda per USB. Segons el primer caràcter:
//   L = trama del guant esquerre, C = calibratge, T = llindars, ? = presentació, X = esborrar calibratge.
static void handleLine(char* s, int len) {
  if (len >= 2 && s[0] == 'L' && s[1] == ',') {                    // trama del guant esquerre (reenviada)
    int body = hwUnseal(s, len);                                    // comprova la suma de verificació
    if (body < 0) { badLines++; return; }
    s[body] = 0;
    char* p = s + 2;
    uint32_t seq = strtoul(p, &p, 10);                              // número de trama del guant esquerre
    if (*p != ',') { badLines++; return; }
    GloveRaw h;
    if (hwParseHand(p + 1, &h) < 0) { badLines++; return; }
    // Si el número no és el següent de l'anterior, compta les trames que s'han perdut pel camí.
    if (leftSeqInit && seq != leftSeq + 1) leftLost += (seq - leftSeq - 1);
    leftSeq = seq; leftSeqInit = true;
    // Guarda la trama (protegida, perquè la tasca dels sensors la llegeix).
    portENTER_CRITICAL(&mux);
    leftRaw = h; leftRxMs = millis(); leftEver = true;
    portEXIT_CRITICAL(&mux);
  } else if (len >= 2 && s[0] == 'C' && s[1] == ',') {            // calibratge
    int body = hwUnseal(s, len);
    if (body < 0) { badLines++; return; }
    s[body] = 0;
    // 22 valors: per a cada mà (primer la dreta), 5 lectures amb la mà oberta,
    // 5 amb el puny tancat i l'angle psi0 de la posició de zero.
    float v[22];
    char* p = s + 2;
    for (int i = 0; i < 22; i++) {
      char* e;
      v[i] = strtof(p, &e);
      if (e == p) { queueLine("{\"type\":\"cal_error\"}\n", 21); return; }   // falta algun valor: error
      p = (*e == ',') ? e + 1 : e;
    }
    for (int h = 0; h < 2; h++) {
      for (int k = 0; k < 5; k++) { cal[h].open[k] = v[h * 11 + k]; cal[h].closed[k] = v[h * 11 + 5 + k]; }
      cal[h].psi0 = v[h * 11 + 10];
    }
    calOk = true;
    validRun = 0;                     // cal tornar a omplir la finestra amb el calibratge nou
    resetDecider = true;
    // Respon a l'aplicació confirmant que ha rebut el calibratge (amb alguns valors per comprovar-lo).
    char b[TX_LEN];
    int n = snprintf(b, sizeof(b), "{\"type\":\"cal_ok\",\"psi0\":[%.6f,%.6f],\"r_open0\":%.2f,\"l_closed4\":%.2f}\n",
                     cal[0].psi0, cal[1].psi0, cal[0].open[0], cal[1].closed[4]);
    queueLine(b, n);
  } else if (len >= 2 && s[0] == 'T' && s[1] == ',') {            // ajustos de la decisió: T,tOn,tRel[,hold,delay]
    int body = hwUnseal(s, len);
    if (body < 0) { badLines++; return; }
    s[body] = 0;
    char* e;
    float on = strtof(s + 2, &e);                                   // nou llindar per ser candidat
    float rel = decider.tRel;                                       // si no s'envien, es mantenen els actuals
    long hold = decider.hold, delay = decider.delay;
    if (*e == ',') rel = strtof(e + 1, &e);                         // nou llindar de "repòs"
    if (*e == ',') hold = strtol(e + 1, &e, 10);                    // nou silenci (finestres)
    if (*e == ',') delay = strtol(e + 1, &e, 10);                   // nova espera (finestres)
    // Només s'accepten valors raonables.
    if (on > 0.2f && on < 1.0f) decider.tOn = on;
    if (rel > 0.2f && rel < 1.0f) decider.tRel = rel;
    if (hold >= 0 && hold <= 20 && delay >= 0 && delay <= 10) lscDeciderSetTiming(&decider, (int)hold, (int)delay);
    resetDecider = true;                                            // comença de nou amb els ajustos nous
    char b[128];
    int n = snprintf(b, sizeof(b), "{\"type\":\"thr_ok\",\"t_on\":%.2f,\"t_rel\":%.2f,\"hold\":%d,\"delay\":%d}\n",
                     decider.tOn, decider.tRel, decider.hold, decider.delay);
    queueLine(b, n);
  } else if (len >= 1 && s[0] == '?') {                            // l'aplicació demana la presentació
    sendHello();
  } else if (len >= 1 && s[0] == 'X') {                            // esborra el calibratge (el model s'atura)
    calOk = false;
  }
}

// S'executa una sola vegada en engegar el guant.
void setup() {
  Serial.setRxBufferSize(4096);       // memòria de recepció gran: hi arriben 50 línies per segon del guant esquerre
  Serial.begin(BAUD);
  delay(200);
  txQ = xQueueCreate(16, sizeof(TxLine));        // cua de sortida de 16 línies
  ringMtx = xSemaphoreCreateMutex();
  // Busca la posició de la classe "repos" a la llista de classes del model.
  for (int i = 0; HAS_MODEL && i < EI_CLASSIFIER_LABEL_COUNT; i++)
    if (strcmp(ei_classifier_inferencing_categories[i], "repos") == 0) restIndex = i;
  lscDeciderInit(&decider, EI_CLASSIFIER_LABEL_COUNT, restIndex, T_ON, T_REL);
  lscDeciderSetTiming(&decider, DEC_HOLD, DEC_DELAY);
  hwBegin();                          // prepara els sensors (el guant ha d'estar quiet)
  sendHello();
  // Crea les dues tasques: sensors al nucli 1 (prioritat alta) i model al nucli 0 (prioritat baixa).
  xTaskCreatePinnedToCore(sensorTask, "sense", 6144, nullptr, 3, nullptr, 1);
#if HAS_MODEL
  xTaskCreatePinnedToCore(inferTask, "infer", 16384, nullptr, 1, nullptr, 0);
#endif
}

// Bucle principal: rep les línies de l'ordinador, envia les línies de la cua de sortida
// i, cada 2 segons, un missatge d'estat.
void loop() {
  static char line[400];              // línia que s'està rebent
  static int len = 0;
  static uint32_t lastStatus = 0;

  // Llegeix caràcter a caràcter fins a trobar un salt de línia; llavors interpreta la línia.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) { line[len] = 0; handleLine(line, len); }
      len = 0;
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = c;
    } else {
      len = 0; badLines++;              // línia massa llarga: es descarta
    }
  }

  // Envia per USB totes les línies que esperen a la cua de sortida.
  TxLine t;
  while (xQueueReceive(txQ, &t, 0) == pdTRUE) Serial.print(t.s);

  // Cada 2 s envia un missatge d'estat: trames, antiguitat i trames perdudes del guant esquerre,
  // línies errònies, calibratge, trames vàlides a la finestra, línies perdudes a la cua,
  // temps de càlcul del model i nombre d'execucions.
  uint32_t now = millis();
  if (now - lastStatus >= 2000) {
    lastStatus = now;
    char b[TX_LEN];
    uint32_t age = leftEver ? now - leftRxMs : 65535;
    int n = snprintf(b, sizeof(b),
                     "{\"type\":\"status\",\"hand\":\"right\",\"frames\":%lu,\"left_age\":%lu,\"left_lost\":%lu,"
                     "\"bad_lines\":%lu,\"cal\":%d,\"valid\":%d,\"tx_drop\":%lu,\"infer_ms\":%lu,\"infers\":%lu}\n",
                     (unsigned long)frameNo, (unsigned long)age, (unsigned long)leftLost, (unsigned long)badLines,
                     calOk ? 1 : 0, validRun > WINDOW_FRAMES ? WINDOW_FRAMES : validRun, (unsigned long)txDrop,
                     (unsigned long)inferMs, (unsigned long)inferCount);
    queueLine(b, n);
  }
  vTaskDelay(1);                      // deixa una mica de temps a les altres tasques
}
