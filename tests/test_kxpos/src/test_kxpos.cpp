#include "utils/PVLog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.hpp>

using PV::kxPos;

int main(int argc, char *argv[]) {
   long kl;

   int nf = 3;
   int nx = 64;
   int ny = 128;

   int nxLocal = nx;
   int nyLocal = ny;

   int inx = nx;
   int iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kxx = (int)(kl % inx);
#else
      int y   = (int)(kl / (nf * nx));
      int kxx = (int)((kl - y * nx * nf) / nf);
#endif
      int kx = kxPos(kl, nxLocal, nyLocal, nf);

      if (kx != kxx) {
         Fatal().printf("FAILED:TEST_KXPOS: (k,kx) = (%ld,%d)\n", kl, kx);
      }
   }

   nx      = 1009;
   ny      = 5;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kxx = (int)(kl % inx);
#else
      int y   = (int)(kl / (nf * nx));
      int kxx = (int)((kl - y * nx * nf) / nf);
#endif
      int kx = kxPos(kl, nxLocal, nyLocal, nf);

      if (kx - kxx != 0) {
         Fatal().printf("FAILED:TEST_KXPOS: (k,kx) = (%ld,%d)\n", kl, kx);
      }
   }

   nf      = 4;
   nx      = 107;
   ny      = 5;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kxx = (int)(kl % inx);
#else
      int y   = (int)(kl / (nf * nx));
      int kxx = (int)((kl - y * nx * nf) / nf);
#endif
      int kx = kxPos(kl, nxLocal, nyLocal, nf);

      if (kx - kxx != 0) {
         Fatal().printf("FAILED:TEST_KXPOS: (k,kx) = (%ld,%d)\n", kl, kx);
      }
   }

   nf      = 1;
   nx      = 16777216 + 1; // this should fail
   nx      = 16777216;
   ny      = 1;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kxx = (int)(kl % inx);
#else
      int y   = (int)(kl / (nf * nx));
      int kxx = (int)((kl - y * nx * nf) / nf);
#endif
      int kx = kxPos(kl, nxLocal, nyLocal, nf);

      if (kx - kxx != 0) {
         Fatal().printf("FAILED:TEST_KXPOS: (k,kx) = (%ld,%d)\n", kl, kx);
      }
   }

   return 0;
}
