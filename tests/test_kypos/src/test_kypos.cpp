#include "utils/PVLog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.hpp>

using PV::kyPos;

int main(int argc, char *argv[]) {
   long kl;

   int nf   = 3;
   int nx = 64;
   int ny = 128;

   int nxLocal = nx;
   int nyLocal = ny;

   int inx = (int)nx;
   int iny = (int)ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (int)((kl / inx) % iny);
#else
      int kk = (int)(kl / (nf * nx));
#endif
      int ky = kyPos(kl, nxLocal, nyLocal, nf);

      if (ky != kk) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%ld,%d)\n", kl, ky);
      }
   }

   nx      = 13;
   ny      = 2007;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (int)((kl / inx) % iny);
#else
      int kk = (int)(kl / (nf * nx));
#endif
      int ky = kyPos(kl, nxLocal, nyLocal, nf);

      if (ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%ld,%d)\n", kl, ky);
      }
   }

   nf      = 4;
   nx      = 5;
   ny      = 107;
   nxLocal = nx;
   nyLocal = ny;

   inx = nx;
   iny = ny;

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (int)((kl / inx) % iny);
#else
      int kk = (int)(kl / (nf * nx));
#endif
      int ky = kyPos(kl, nxLocal, nyLocal, nf);

      if (ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%ld,%d)\n", kl, ky);
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

   for (kl = 0; kl < (long)inx * (long)iny * (long)nf; kl++) {
#ifdef FEATURES_LAST
      int kk = (int)((kl / inx) % iny);
#else
      int kk = (int)(kl / (nf * nx));
#endif
      int ky = kyPos(kl, nxLocal, nyLocal, nf);

      if (ky - kk != 0) {
         Fatal().printf("FAILED:TEST_KYPOS: (k,ky) = (%ld,%d)\n", kl, ky);
      }
   }

   return 0;
}
