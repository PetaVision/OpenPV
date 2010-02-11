/**
 * This file tests copying to boundary regions while applying mirror boundary conditions.
 * formerly called test_borders.cpp
 *
 */

#undef DEBUG_PRINT

#include "../src/layers/HyPerLayer.hpp"
#include "../src/layers/Example.hpp"
#include "../src/io/io.h"


const int numFeatures = 2;

int main(int argc, char * argv[])
{
   PVLayerLoc sLoc, bLoc;
   PVLayerCube * sCube, * bCube;

   PV::HyPerCol * hc = new PV::HyPerCol("test_mirror_BCs column", argc, argv);
   PV::Example * l = new PV::Example("test_mirror_BCs layer", hc);

   //FILE * fd = stdout;
   int nf  = numFeatures;
   int nB = 4;
   int nS = 8;
   int syex = ( nS + 2*nB ) * nf;
   int sy = nS * nf;

   sLoc.nxGlobal = sLoc.nyGlobal = nS; // shouldn't be used
   sLoc.kx0 = sLoc.ky0 = 0; // shouldn't be used
   sLoc.nx = sLoc.ny = nS;
   sLoc.nPad = nB;
   sLoc.nBands = nf;

   bLoc = sLoc;

   sCube = pvcube_new(&sLoc, (nS+2*nB)*(nS+2*nB)*nf);
   bCube = sCube;

   // fill interior with non-extended index of each neuron
   // leave border values at zero to start with
   int kFirst = nB;
   int kLast = nS + nB;
   for (int ky = kFirst; ky < kLast; ky++) {
      for (int kx = kFirst; kx < kLast; kx++) {
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            sCube->data[kex] = k;
            printf("sCube val = %5i:, kex = %5i:, k = %5i\n", (int) sCube->data[kex], kex, k);
        }
      }
   }

   // this is the function we're testing...
   for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
      l->mirrorInteriorToBorder(borderId, sCube, bCube);
   }

   // write out extended cube values
   int nx = 2*nB + nS;
   int ny = nx;
   for (int kf = 0; kf < nf; kf++) {
      for (int ky = 0; ky < ny; ky++) {
         for (int kx = 0; kx < nx; kx++) {
            int kex = ky * syex + kx * nf + kf;
            printf("%5i ", (int) sCube->data[kex]);
         }
         printf("\n");
      }
      printf("\n");
   }

   // check values at mirror indices
   // uses a completely different algorithm than mirrorInteriorToBorder
   // northwest
   for (int ky = kFirst; ky < kFirst+nB; ky++) {
      int kymirror = nB - (ky - nB) - 1;
      for (int kx = kFirst; kx < kFirst+nB; kx++) {
         int kxmirror = nB - (kx - nB) - 1;
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   pvcube_delete(sCube);
   sCube = bCube = NULL;

   delete hc;

   return 0;
}
