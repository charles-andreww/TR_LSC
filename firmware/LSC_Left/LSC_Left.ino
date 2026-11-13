
// Què fa aquest programa:
//   * Llegeix els sensors del guant 100 vegades per segon.
//   * Cada 20 ms (50 vegades per segon) envia una línia de text per USB a l'ordinador:
//         L,seq,f0..f4,qw,qx,qy,qz,gx,gy,gz,ax,ay,az,flags*HH
//     (L = guant esquerre, seq = número de trama, després els 16 valors de la mà
//      i, al final, la suma de verificació).
//   * L'aplicació de l'ordinador (lsc_app.py) reenvia cada línia al guant dret,
//     que és el que executa el model d'intel·ligència artificial.
//   * No fa servir el Wi-Fi ni cap ràdio: en activar-la, la placa es reiniciava
//     per caigudes de tensió (brownout).
// Carpeta: LSC_Left.ino + glove_hw.h
// En engegar-lo: cal tenir el guant QUIET durant 2 s (calibratge del giroscopi).


// Pins de l'ESP32 on hi ha connectat cada dit: polze, índex, cor, anular, petit
// (els mateixos que al programa de prova del guant).
static const int GLOVE_PINS[5] = {34, 35, 32, 33, 39};

#include "glove_hw.h"            // lectura dels sensors (comú als dos guants)

#define BAUD 460800              // velocitat de la comunicació USB

static uint32_t seqNo = 0;       // número de trama: augmenta en 1 a cada línia enviada

// Envia un missatge de presentació: quin guant és, la versió del programa i si la IMU
// i el calibratge del giroscopi han funcionat. L'aplicació el fa servir per reconèixer el guant.
static void sendHello() {
  Serial.printf("{\"type\":\"hello\",\"hand\":\"left\",\"fw\":\"2.0\",\"imu\":%d,\"bias_ok\":%d}\n",
                hwImuOk ? 1 : 0, hwBiasOk ? 1 : 0);
}

// S'executa una sola vegada en engegar el guant.
void setup() {
  Serial.begin(BAUD);            // obre la comunicació USB
  delay(200);
  hwBegin();                     // prepara els sensors (el guant ha d'estar quiet)
  sendHello();
}

// S'executa contínuament, una vegada cada 10 ms.
void loop() {
  static TickType_t last = xTaskGetTickCount();   // moment de l'última volta (per mantenir el ritme)
  static uint8_t tick = 0;                         // comptador de voltes (0 o 1)

  // Si l'ordinador envia '?', el guant torna a enviar el missatge de presentació.
  while (Serial.available()) {
    int c = Serial.read();
    if (c == '?') sendHello();
  }

  hwTick();                              // llegeix els sensors (cada 10 ms)
  if (++tick >= 2) {                     // una de cada dues voltes = cada 20 ms
    tick = 0;
    GloveRaw h;
    hwTakeFrame(&h);                     // agafa les dades de la trama
    char buf[200];
    // Construeix la línia: "L,número," + els 16 valors + "*HH\n" i l'envia per USB.
    int n = snprintf(buf, sizeof(buf), "L,%lu,", (unsigned long)seqNo++);
    n += hwFormatHand(buf + n, sizeof(buf) - n, &h);
    n = hwSeal(buf, n, sizeof(buf));
    if (n > 0) Serial.write((const uint8_t*)buf, n);
  }

  vTaskDelayUntil(&last, pdMS_TO_TICKS(HW_TICK_MS));   // espera fins a completar els 10 ms exactes
}
