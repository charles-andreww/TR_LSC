// Aquest fitxer és idèntic als dos guants (LSC_Left i LSC_Right). Conté tot el que
// fa falta per llegir els sensors d'un guant:
//
//   * Ritme de treball: la funció hwTick() s'executa cada 10 ms (100 vegades per
//     segon) i llegeix els sensors. Cada 20 ms, hwTakeFrame() agrupa el que s'ha
//     llegit en una "trama" (50 trames per segon).
//
//   * Sensors de flexió (un per dit): cada dit es llegeix 12 vegades seguides, es
//     descarten les 3 lectures més baixes i les 3 més altes, i es fa la mitjana de
//     les 6 del mig (així s'eliminen els pics de soroll). Després el valor es
//     suavitza amb una mitjana mòbil exponencial (factor 0,35), que endarrereix el
//     senyal només uns 23 ms. El resultat són comptes de l'ADC (0 a 4095), sense
//     convertir encara a "grau de flexió".
//
//   * IMU (MPU-6050, acceleròmetre + giroscopi): configurada a ±1000 graus/s i ±4 g
//     (la versió anterior feia servir ±250 graus/s, i un moviment ràpid de la mà
//     saturava el sensor). S'activa el filtre passa-baixos intern de 44 Hz. Amb un
//     filtre de Mahony es calcula l'orientació de la mà (un quaternió) 100 vegades
//     per segon. L'acceleròmetre només s'utilitza per corregir l'orientació quan
//     mesura aproximadament 1 g, és a dir, quan la mà no està accelerant; així els
//     signes ràpids no deformen l'orientació calculada.
//     L'error de zero del giroscopi (biaix) es mesura en engegar el guant (cal
//     tenir-lo quiet uns 1,5 s) i es va actualitzant cada vegada que el guant està quiet.
//
//   * Contingut de cada trama: l'orientació (quaternió, del guant cap al món, amb
//     l'eix Z del món apuntant amunt), la velocitat de gir (rad/s) i l'acceleració
//     lineal (en g, ja sense la gravetat). Gir i acceleració són la mitjana de les
//     dues lectures que hi ha dins de cada trama, en els eixos del guant.
//
// Una mà s'envia com a 16 valors separats per comes:
//   f0..f4 (5 dits), qw,qx,qy,qz (quaternió), gx,gy,gz (gir), ax,ay,az (acceleració), flags
//   flags: bit 0 = la IMU funciona, bit 1 = el guant està quiet
// Cada línia acaba amb *HH, on HH és una suma de verificació (XOR de tots els
// caràcters anteriors a '*'). Serveix per detectar caràcters corromputs pel cable USB.
//
// Abans d'incloure aquest fitxer, el programa de cada guant ha de definir
// GLOVE_PINS[5]: els pins on hi ha connectats els dits (polze, índex, cor, anular, petit).

#pragma once            // evita que el fitxer s'inclogui dues vegades
#include <Arduino.h>
#include <Wire.h>       // biblioteca del bus I2C (per parlar amb la IMU)

#define HW_TICK_MS   10     // període de lectura dels sensors: 10 ms (100 Hz)
#define MPU_ADDR     0x68   // adreça I2C de la IMU MPU-6050
#define GLOVE_SDA    22     // pin de dades del bus I2C
#define GLOVE_SCL    21     // pin de rellotge del bus I2C
#define FLAG_IMU     1      // indicador: la IMU funciona
#define FLAG_STILL   2      // indicador: el guant està quiet
#define FLAG_LEFT_OK 4      // indicador (només al guant dret): les dades del guant esquerre arriben bé

// Dades "en cru" d'una mà en una trama: 5 dits, quaternió (4), gir (3), acceleració (3) i indicadors.
struct GloveRaw { float f[5]; float q[4]; float g[3]; float a[3]; unsigned flags; };

// variables d'estat
static float    hwFlex[5];                       // valor suavitzat de cada dit (comptes de l'ADC)
static bool     hwImuOk = false, hwBiasOk = false, hwStill = false;  // IMU funciona / biaix mesurat bé / guant quiet
static float    hwQ[4] = {1, 0, 0, 0};           // orientació actual (quaternió); {1,0,0,0} = sense rotació
static float    hwBias[3] = {0, 0, 0};           // biaix del giroscopi a cada eix, en unitats del sensor (LSB)
static float    hwSumG[3], hwSumA[3];            // sumes de gir i acceleració dins la trama actual
static int      hwSumN = 0;                      // quantes lectures s'han sumat a la trama actual
static uint32_t hwStillTicks = 0;                // quantes lectures seguides fa que el guant està quiet
static uint32_t hwLastUs = 0;                    // moment de l'última lectura (microsegons)

static const float HW_GYRO_LSB = 32.8f;          // unitats del sensor per cada grau/s (amb el rang de ±1000 graus/s)
static const float HW_ACC_LSB  = 8192.0f;        // unitats del sensor per cada g (amb el rang de ±4 g)
static const float HW_TWO_KP   = 1.0f;           // força amb què l'acceleròmetre corregeix l'orientació (filtre de Mahony)

//funcions per al bus I2C
// Escriu el valor 'val' al registre 'reg' de la IMU. Retorna true si la IMU ha respost bé.
static bool hwWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission(true) == 0;
}
//Llegeix d'una sola vegada els 14 registres de mesura de la IMU, a partir del registre 0x3B:
// acceleració (6 bytes), temperatura (2 bytes, no es fa servir) i gir (6 bytes).
// Cada valor ocupa 2 bytes (primer el byte alt). Retorna false si la lectura falla.
static bool hwReadImu(int16_t acc[3], int16_t gyr[3]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 14, true) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  for (int i = 0; i < 3; i++) acc[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]);       // bytes 0-5: acceleració
  for (int i = 0; i < 3; i++) gyr[i] = (int16_t)((b[8 + 2 * i] << 8) | b[9 + 2 * i]);   // bytes 8-13: gir
  return true;
}

//sensors de flexió
// Llegeix un dit: 12 lectures de l'ADC, les ordena de menor a major, descarta les 3 més
// baixes i les 3 més altes i retorna la mitjana de les 6 del mig ("mitjana retallada").
// Així una lectura aïllada amb un pic de soroll no afecta el resultat.
static float hwReadFinger(int pin) {
  int v[12];
  for (int s = 0; s < 12; s++) v[s] = analogRead(pin);
  for (int i = 1; i < 12; i++) {                 // ordenació per inserció (de menor a major)
    int k = v[i], j = i - 1;
    while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
    v[j + 1] = k;
  }
  float s = 0;
  for (int i = 3; i < 9; i++) s += v[i];         // posicions 3 a 8: les 6 lectures centrals
  return s / 6.0f;
}

//filtre de Mahony
// Actualitza l'orientació hwQ a partir de l'acceleració (ax, ay, az, en g), el gir
//(gx, gy, gz, en rad/s) i el temps transcorregut dt (en segons).
// Idea: el giroscopi diu com ha girat la mà, però a la llarga acumula error. L'acceleròmetre,
// quan la mà està quieta, indica on és "avall" (la gravetat) i serveix per corregir aquest error.
static void hwMahony(float ax, float ay, float az, float gx, float gy, float gz, float dt) {
  float q0 = hwQ[0], q1 = hwQ[1], q2 = hwQ[2], q3 = hwQ[3];
  float an = sqrtf(ax * ax + ay * ay + az * az);           // mòdul de l'acceleració
  if (an > 0.85f && an < 1.15f) {                // només es confia en la gravetat si la mà no accelera (mòdul prop d'1 g)
    ax /= an; ay /= an; az /= an;                // direcció de la gravetat mesurada (vector de longitud 1)
    // Direcció on hauria d'estar la gravetat segons l'orientació calculada fins ara.
    float vx = 2.0f * (q1 * q3 - q0 * q2);
    float vy = 2.0f * (q0 * q1 + q2 * q3);
    float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
    // Error = producte vectorial entre la gravetat mesurada i la calculada.
    // Es suma al gir perquè l'orientació es vagi acostant a la correcta.
    gx += HW_TWO_KP * (ay * vz - az * vy);
    gy += HW_TWO_KP * (az * vx - ax * vz);
    gz += HW_TWO_KP * (ax * vy - ay * vx);
  }
  // Fa girar el quaternió segons la velocitat de gir (corregida) durant el temps dt.
  float qa = q0, qb = q1, qc = q2;
  q0 += 0.5f * (-qb * gx - qc * gy - q3 * gz) * dt;
  q1 += 0.5f * ( qa * gx + qc * gz - q3 * gy) * dt;
  q2 += 0.5f * ( qa * gy - qb * gz + q3 * gx) * dt;
  q3 += 0.5f * ( qa * gz + qb * gy - qc * gx) * dt;
  // Torna a fer que el quaternió tingui longitud 1 (els petits errors de càlcul el deformarien).
  float n = 1.0f / sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  hwQ[0] = q0 * n; hwQ[1] = q1 * n; hwQ[2] = q2 * n; hwQ[3] = q3 * n;
}

// Orientació inicial: a partir de la gravetat mesurada amb el guant quiet, calcula la
// inclinació de la mà (balanceig i capcineig) i la converteix en quaternió. La direcció
// horitzontal (guinyada) comença a zero, perquè l'acceleròmetre no la pot mesurar.
static void hwQuatFromAccel(float ax, float ay, float az) {
  float roll = atan2f(ay, az), pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
  float cr = cosf(roll / 2), sr = sinf(roll / 2), cp = cosf(pitch / 2), sp = sinf(pitch / 2);
  hwQ[0] = cr * cp; hwQ[1] = sr * cp; hwQ[2] = cr * sp; hwQ[3] = -sr * sp;
}

// posada en marxa
// Prepara l'ADC i la IMU i mesura el biaix del giroscopi.
// El guant ha d'estar QUIET durant l'estona que dura (uns 1,5 s).
static void hwBegin() {
  analogReadResolution(12);                      // l'ADC dona valors de 12 bits (0 a 4095)
  analogSetAttenuation(ADC_11db);                // atenuació d'11 dB: permet mesurar tensions de fins a uns 3,1 V
  for (int f = 0; f < 5; f++) hwFlex[f] = hwReadFinger(GLOVE_PINS[f]);   // primer valor de cada dit

  Wire.begin(GLOVE_SDA, GLOVE_SCL, 400000);      // engega el bus I2C a 400 kHz
  Wire.setTimeOut(25);                           // si la IMU no respon en 25 ms, la lectura falla
  // Configuració de la IMU (escriptura a diversos registres). Si alguna falla, la IMU no es fa servir.
  hwImuOk = hwWrite(0x6B, 0x01)          // desperta la IMU i fa servir el rellotge del giroscopi
         && hwWrite(0x19, 0x00)          // divisor de la freqüència de mostreig = 0
         && hwWrite(0x1A, 0x03)          // filtre passa-baixos intern de 44 Hz
         && hwWrite(0x1B, 0x10)          // giroscopi a ±1000 graus/s
         && hwWrite(0x1C, 0x08);         // acceleròmetre a ±4 g
  if (!hwImuOk) return;
  delay(50);                                     // deixa que la IMU s'estabilitzi

  // Mesura del biaix del giroscopi: fins a 3 intents.
  for (int attempt = 0; attempt < 3 && !hwBiasOk; attempt++) {
    // s = suma del gir, mn/mx = valor mínim i màxim de cada eix, sa = suma de l'acceleració
    float s[3] = {0, 0, 0}, mn[3] = {1e9, 1e9, 1e9}, mx[3] = {-1e9, -1e9, -1e9}, sa[3] = {0, 0, 0};
    int n = 0;                                   // lectures correctes
    for (int i = 0; i < 200; i++) {              // 200 lectures, una cada 5 ms (1 segon en total)
      int16_t a[3], g[3];
      if (hwReadImu(a, g)) {
        for (int k = 0; k < 3; k++) {
          s[k] += g[k]; sa[k] += a[k];
          if (g[k] < mn[k]) mn[k] = g[k];
          if (g[k] > mx[k]) mx[k] = g[k];
        }
        n++;
      }
      delay(5);
    }
    if (n < 100) { hwImuOk = false; return; }   // menys de la meitat de lectures bones: la IMU falla
    // Si el gir ha variat més de ~3 graus/s, el guant s'ha mogut i la mesura no és fiable.
    bool still = true;
    for (int k = 0; k < 3; k++) if (mx[k] - mn[k] > 3.0f * HW_GYRO_LSB) still = false;   // s'ha mogut més de ~3 graus/s
    for (int k = 0; k < 3; k++) hwBias[k] = s[k] / n;              // biaix = mitjana del gir en repòs
    hwQuatFromAccel(sa[0] / n, sa[1] / n, sa[2] / n);              // orientació inicial a partir de la gravetat
    hwBiasOk = still;                            // si s'ha mogut, es torna a intentar
  }
  hwLastUs = micros();
}

// lectura a 100 Hz
// S'executa cada 10 ms: llegeix els dits i la IMU, actualitza l'orientació i
// acumula el gir i l'acceleració per a la trama.
static void hwTick() {
  // Dits: mitjana mòbil exponencial. El nou valor s'acosta un 35 % cap a la lectura nova.
  for (int f = 0; f < 5; f++) hwFlex[f] += 0.35f * (hwReadFinger(GLOVE_PINS[f]) - hwFlex[f]);
  if (!hwImuOk) return;

  int16_t ra[3], rg[3];
  // Temps real transcorregut des de l'última lectura (en segons).
  uint32_t now = micros();
  float dt = (now - hwLastUs) * 1e-6f;
  hwLastUs = now;
  if (dt <= 0.0f || dt > 0.1f) dt = HW_TICK_MS / 1000.0f;   // valor estrany, es fan servir els 10 ms previstos
  if (!hwReadImu(ra, rg)) return;

  // Conversió a unitats físiques: acceleració en g i gir en rad/s (restant-hi el biaix).
  float a[3], g[3];
  for (int k = 0; k < 3; k++) {
    a[k] = ra[k] / HW_ACC_LSB;
    g[k] = (rg[k] - hwBias[k]) / HW_GYRO_LSB * (PI / 180.0f);
  }
  // Detecció de guant quiet: acceleració molt propera a 1 g i gir inferior a 1,5 graus/s.
  float an = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  float gn = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  bool stillNow = fabsf(an - 1.0f) < 0.03f && gn < 0.026f;       // gir < 1,5 graus/s
  hwStillTicks = stillNow ? hwStillTicks + 1 : 0;
  hwStill = hwStillTicks >= 50;                                     // quiet durant almenys 0,5 s (50 lectures)
  if (hwStill)                                                      // si està quiet, el biaix s'actualitza a poc a poc
    for (int k = 0; k < 3; k++) hwBias[k] += 0.002f * (rg[k] - hwBias[k]);

  hwMahony(a[0], a[1], a[2], g[0], g[1], g[2], dt);              // actualitza l'orientació

  // Treu la gravetat de l'acceleració: es calcula on apunta la gravetat segons l'orientació
  // i es resta, de manera que només queda l'acceleració deguda al moviment de la mà.
  float q0 = hwQ[0], q1 = hwQ[1], q2 = hwQ[2], q3 = hwQ[3];
  float grav[3] = {2.0f * (q1 * q3 - q0 * q2), 2.0f * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3};
  for (int k = 0; k < 3; k++) { hwSumG[k] += g[k]; hwSumA[k] += a[k] - grav[k]; }   // acumula per a la trama
  hwSumN++;
}

//  trama a 50 Hz
// Omple 'o' amb les dades de la trama actual (cada 20 ms) i torna a començar les sumes.
static void hwTakeFrame(GloveRaw* o) {
  for (int f = 0; f < 5; f++) o->f[f] = hwFlex[f];                 // dits: valor suavitzat més recent
  for (int k = 0; k < 4; k++) o->q[k] = hwQ[k];                    // orientació més recent
  for (int k = 0; k < 3; k++) {                                    // gir i acceleració: mitjana de la trama
    o->g[k] = hwSumN ? hwSumG[k] / hwSumN : 0.0f;
    o->a[k] = hwSumN ? hwSumA[k] / hwSumN : 0.0f;
    hwSumG[k] = hwSumA[k] = 0.0f;
  }
  hwSumN = 0;
  o->flags = (hwImuOk ? FLAG_IMU : 0) | (hwStill ? FLAG_STILL : 0);
}

// línies de text
// Suma de verificació, combina tots els caràcters amb l'operació XOR.
// Si un caràcter canvia pel camí, el resultat (gairebé sempre) canvia i la línia es descarta.
static uint8_t hwXor(const char* s, int n) {
  uint8_t x = 0;
  for (int i = 0; i < n; i++) x ^= (uint8_t)s[i];
  return x;
}
// "Segella" una línia: hi afegeix "*HH\n" (HH = suma de verificació en hexadecimal).
// n = longitud actual de la línia, cap = mida del buffer. Retorna la longitud nova (-1 si no hi cap).
static int hwSeal(char* buf, int n, int cap) {
  if (n < 0 || n + 4 >= cap) return -1;
  return n + snprintf(buf + n, cap - n, "*%02X\n", hwXor(buf, n));
}
// Escriu els 16 valors d'una mà com a text separat per comes (sense coma al principi ni al final).
// Dits amb 1 decimal, quaternió amb 5, gir i acceleració amb 4. Retorna la longitud escrita.
static int hwFormatHand(char* buf, int cap, const GloveRaw* h) {
  return snprintf(buf, cap, "%.1f,%.1f,%.1f,%.1f,%.1f,%.5f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%u",
                  h->f[0], h->f[1], h->f[2], h->f[3], h->f[4], h->q[0], h->q[1], h->q[2], h->q[3],
                  h->g[0], h->g[1], h->g[2], h->a[0], h->a[1], h->a[2], h->flags);
}
// Comprova el "*HH" del final d'una línia rebuda (len = longitud sense el salt de línia).
// Si la suma de verificació coincideix, retorna la longitud del contingut; si no, -1.
static int hwUnseal(const char* line, int len) {
  if (len < 4 || line[len - 3] != '*') return -1;
  unsigned v = 0;
  if (sscanf(line + len - 2, "%2X", &v) != 1) return -1;
  return hwXor(line, len - 3) == v ? len - 3 : -1;
}
// Llegeix els 16 valors d'una mà a partir del text "v0,v1,...,v15" i els guarda a 'h'.
// Retorna quants caràcters ha llegit, o -1 si el text no té el format correcte.
static int hwParseHand(const char* s, GloveRaw* h) {
  int used = 0;
  float v[16];
  for (int i = 0; i < 16; i++) {
    char* end;
    v[i] = strtof(s + used, &end);               // converteix el text següent en un nombre
    if (end == s + used) return -1;              // no hi havia cap nombre: error
    used = end - s;
    if (i < 15) { if (s[used] != ',') return -1; used++; }   // entre valors hi ha d'haver una coma
  }
  for (int k = 0; k < 5; k++) h->f[k] = v[k];
  for (int k = 0; k < 4; k++) h->q[k] = v[5 + k];
  for (int k = 0; k < 3; k++) { h->g[k] = v[9 + k]; h->a[k] = v[12 + k]; }
  h->flags = (unsigned)v[15];
  return used;
}
