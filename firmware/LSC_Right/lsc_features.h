
// Aquest càlcul és una còpia, línia per línia, de la funció hand_features() del fitxer
// pc/lsc_core.py, que és la que fa servir build_dataset.py per preparar les dades
// d'entrenament. Així el model rep en temps real exactament el mateix tipus de números
// amb què s'ha entrenat. La prova tests/test_c_vs_python.py compila AQUEST fitxer a
// l'ordinador i comprova que les dues versions donen els mateixos resultats.
// Si se'n canvia una, cal canviar també l'altra.
//
// Per a cada mà es calculen 17 valors:
//   0..4   flexió de cada dit: (obert - lectura) / (obert - tancat), limitada entre -0,2 i 1,2
//          (0 = dit estirat, 1 = puny tancat)
//   5..7   eix X del guant (cap on apunten els dits)       } expressats en el sistema del cos:
//   8..10  eix Z del guant (perpendicular al palmell)      } X = endavant (cap on apuntaven les mans
//                                                          }     en la posició de zero), Y = esquerra,
//                                                          }     Z = amunt (contrari a la gravetat)
//   11..13 velocitat de gir: rad/s dividits per 8, en els eixos del guant
//   14..16 acceleració sense la gravetat, en el sistema del cos, en g dividits per 2
// Les divisions per 8 i per 2 fan que tots els valors siguin de l'ordre de la unitat.
// C pur, sense biblioteques d'Arduino (per poder-lo provar a l'ordinador).

#pragma once
#include <math.h>

#define LSC_FEAT_PER_HAND 17      // valors per mà
#define LSC_N_FEAT        34      // valors totals (dues mans)
#define LSC_GYRO_SCALE    8.0f    // el gir es divideix per 8
#define LSC_ACC_SCALE     2.0f    // l'acceleració es divideix per 2
#define LSC_FLEX_MIN      (-0.2f) // límit inferior de la flexió normalitzada
#define LSC_FLEX_MAX      1.2f    // límit superior de la flexió normalitzada

// Calibratge d'una mà: lectura de cada dit amb la mà oberta i amb el puny tancat,
// i psi0, l'angle horitzontal cap on apuntaven els dits en la posició de zero.
typedef struct { float open[5]; float closed[5]; float psi0; } LscCalHand;
// Dades en cru d'una mà (les mateixes que GloveRaw de glove_hw.h).
typedef struct { float f[5]; float q[4]; float g[3]; float a[3]; unsigned flags; } LscRawHand;

// Calcula els 17 valors d'una mà a partir de les dades en cru 'r' i del calibratge 'c'.
static inline void lscHandFeatures(const LscRawHand* r, const LscCalHand* c, float out[LSC_FEAT_PER_HAND]) {
  // Flexió dels dits: com més doblegat, més baixa és la lectura, per això es fa (obert - lectura).
  for (int i = 0; i < 5; i++) {
    float b = (c->open[i] - r->f[i]) / (c->open[i] - c->closed[i]);
    if (b < LSC_FLEX_MIN) b = LSC_FLEX_MIN;
    if (b > LSC_FLEX_MAX) b = LSC_FLEX_MAX;
    out[i] = b;
  }
  // Orientació: el quaternió es normalitza (longitud 1) i es converteix en una matriu de
  //    rotació R de 3x3. Cada columna de R és un eix del guant vist des del sistema fix.
  float n = sqrtf(r->q[0] * r->q[0] + r->q[1] * r->q[1] + r->q[2] * r->q[2] + r->q[3] * r->q[3]);
  if (n < 1e-9f) n = 1.0f;                       // evita dividir per zero
  float w = r->q[0] / n, x = r->q[1] / n, y = r->q[2] / n, z = r->q[3] / n;
  float R[3][3] = {
    {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)},
    {2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
    {2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)}};
  // Gir al voltant de l'eix vertical: Rb = Rz(-psi0) * R. Així "endavant" passa a ser
  //    la direcció on apuntaven els dits en la posició de zero, sigui quina sigui la
  //    direcció en què s'ha engegat el guant.
  float cs = cosf(c->psi0), sn = sinf(c->psi0);
  float Rb[3][3];
  for (int j = 0; j < 3; j++) {               // Rb = Rz(-psi0) * R
    Rb[0][j] =  cs * R[0][j] + sn * R[1][j];
    Rb[1][j] = -sn * R[0][j] + cs * R[1][j];
    Rb[2][j] =  R[2][j];                      // l'eix vertical no canvia
  }
  for (int i = 0; i < 3; i++) {
    out[5 + i] = Rb[i][0];                    // primera columna: eix X del guant (cap on apunten els dits)
    out[8 + i] = Rb[i][2];                    // tercera columna: eix Z del guant (perpendicular al palmell)
    out[11 + i] = r->g[i] / LSC_GYRO_SCALE;   // velocitat de gir
    // Acceleració: es passa dels eixos del guant als del cos (multiplicant per Rb).
    out[14 + i] = (Rb[i][0] * r->a[0] + Rb[i][1] * r->a[1] + Rb[i][2] * r->a[2]) / LSC_ACC_SCALE;
  }
}
