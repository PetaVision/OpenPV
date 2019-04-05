#include "utils/PVLog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.hpp>

int main(int argc, char *argv[]) {
   int kl;

   int nf   = 3;
   float nx = 64;
   float ny = 128;

   float nxLocal = nx;
   float nyLocal = ny;

   int inx = (int)nx;
   int iny = (int)ny;

   for (kl = 0; kl < inx * iny * nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (kl / inx) % iny;
#else
      int kk = kl / (nf * nx);
#endif
      float ky = kyPos(kl, nxLocal, nyLocal, nf);

      if (ky != (float)kk) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%d,%f)\n", kl, (double)ky);
      }
   }

   nx      = 13;
   ny      = 2007;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < inx * iny * nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (kl / inx) % iny;
#else
      int kk = kl / (nf * nx);
#endif
      float ky = kyPos(kl, nxLocal, nyLocal, nf);

      if ((int)ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%d,%f)\n", kl, (double)ky);
      }
   }

   nf      = 4;
   nx      = 5;
   ny      = 107;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < inx * iny * nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (kl / inx) % iny;
#else
      int kk = kl / (nf * nx);
#endif
      float ky = kyPos(kl, nxLocal, nyLocal, nf);

      if ((int)ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%d,%f)\n", kl, (double)ky);
      }
   }

   nf      = 1;
   nx      = 1;
   ny      = 16777216 + 2; // this should fail for FEATURES_LAST
   ny      = 16777216;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < inx * iny * nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (kl / inx) % iny;
#else
      int kk = kl / (nf * nx);
#endif
      float ky = kyPos(kl, nxLocal, nyLocal, nf);

      if ((int)ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%d,%f)\n", kl, (double)ky);
      }
   }

   return 0;
}
