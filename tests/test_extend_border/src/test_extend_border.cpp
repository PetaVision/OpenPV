#include "utils/PVLog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.hpp>

using PV::kxPos;
using PV::kyPos;
using PV::featureIndex;
using PV::kIndex;
using PV::kIndexExtended;

//
// A replacement for globalIndexFromLocal from conversions.hpp.
// WARNING - any changes in conversions.hpp should be reflected here.
static inline long globalIndexFromLocal_nompi(long kl, PVLayerLoc loc) {
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int kf  = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
}

int main(int argc, char *argv[]) {
   int nf = 3;

   int nx = 64;
   int ny = 68;
   int nb = 4;

   int nxGlobal = nx + 2 * nb;
   int nyGlobal = ny + 2 * nb;

   int kx0 = nb;
   int ky0 = nb;

   PVLayerLoc loc;
   loc.nx       = nx;
   loc.ny       = ny;
   loc.nxGlobal = nxGlobal;
   loc.nyGlobal = nyGlobal;
   loc.kx0      = kx0;
   loc.ky0      = ky0;
   loc.halo.lt  = nb;
   loc.halo.rt  = nb;
   loc.halo.dn  = nb;
   loc.halo.up  = nb;
   loc.nf       = nf;

   long nGlobal = (long)nf * (long)nxGlobal * (long)nyGlobal;
   for (long kl = 0; kl < nGlobal; kl++) {
      long kg = globalIndexFromLocal_nompi(kl, loc);
      long kb = kIndexExtended(
            kl, nx, ny, nf, nb, nb, nb, nb); // All margin widths the same.  Should generalize
      if (kb != kg) {
         Fatal().printf("FAILED:TEST_EXTEND_BORDER: (kl,kb) = (%d,%d)\n", kl, kb);
      }
   }

   return 0;
}
