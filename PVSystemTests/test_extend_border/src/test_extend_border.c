#include <utils/conversions.h>
#include <stdio.h>
#include <stdlib.h>

//
// A replacement for globalIndexFromLocal from conversions.h.
// WARNING - any changes in conversions.h should be reflected here.
static inline int globalIndexFromLocal_nompi(int kl, PVLayerLoc loc)
{
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int  kf = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
}

int main(int argc, char* argv[])
{
   int kg, kl, kb;

   PVLayerLoc loc;

   float nf = 3;

   float nx = 64.0;
   float ny = 68.0;
   float nb = 4.0;

   float nxGlobal = nx + 2*nb;
   float nyGlobal = ny + 2*nb;

   float kx0 = nb;
   float ky0 = nb;

   loc.nx = nx;
   loc.ny = ny;
   loc.nxGlobal = nxGlobal;
   loc.nyGlobal = nyGlobal;
   loc.kx0 = kx0;
   loc.ky0 = ky0;
   loc.halo.lt  = nb;
   loc.halo.rt  = nb;
   loc.halo.dn  = nb;
   loc.halo.up  = nb;
   loc.nf  = nf;

   for (kl = 0; kl < nf*nxGlobal*nyGlobal; kl++) {
      kg = globalIndexFromLocal_nompi(kl, loc);
      kb = kIndexExtended(kl, nx, ny, nf, nb, nb, nb, nb); // All margin widths the same.  Should generalize
      if (kb != kg) {
         printf("FAILED:TEST_EXTEND_BORDER: (kl,kb) = (%d,%d)\n", kl, kb);
         exit(1);
      }
   }

   return 0;
}
