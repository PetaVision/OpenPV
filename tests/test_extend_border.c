#include "../src/utils/conversions.h"
#include "../src/layers/PVLayer.h"
#include <stdio.h>
#include <stdlib.h>

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
   loc.nPad = nb;
   loc.nBands = nf;

   for (kl = 0; kl < nf*nxGlobal*nyGlobal; kl++) {
      kg = globalIndexFromLocal(kl, loc, nf);
      kb = kIndexExtended(kl, nx, ny, nf, nb);
      if (kb != kg) {
	 printf("FAILED:TEST_EXTEND_BORDER: (kl,kb) = (%d,%d)\n", kl, kb);
	 exit(1);
      }
   }

  return 0;
}
