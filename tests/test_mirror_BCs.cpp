/**
 * This file tests copying to boundary regions while applying mirror boundary conditions.
 * formerly called test_borders.cpp
 *
 */

#undef DEBUG_PRINT

#include "Example.hpp"
#include "../src/layers/HyPerLayer.hpp"
#include "../src/io/io.h"


//const int numFeatures = 1;

int main(int argc, char * argv[])
{
   char * cl_args[3];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/test_mirror_BCs.params");
   PVLayerLoc sLoc, bLoc;
   PVLayerCube * sCube, * bCube;

   PV::HyPerCol * hc = new PV::HyPerCol("test_mirror_BCs column", 3, cl_args);
   PV::Example * l = new PV::Example("test_mirror_BCs layer", hc);

   //FILE * fd = stdout;
   int nf = l->clayer->loc.nf;
   int nB = l->clayer->loc.nb; //4;
   int nS = l->clayer->loc.nx; // 8;
   int syex = ( nS + 2*nB ) * nf;
   int sy = nS * nf;
   int nx = 2*nB + nS;
   int ny = nx;

   sLoc.nxGlobal = sLoc.nyGlobal = nS; // shouldn't be used
   sLoc.kx0 = sLoc.ky0 = 0; // shouldn't be used
   sLoc.nx = sLoc.ny = nS;
   sLoc.nb = nB;
   sLoc.nf = nf;
   sLoc.halo.lt = l->getLayerLoc()->halo.lt;
   sLoc.halo.rt = l->getLayerLoc()->halo.rt;
   sLoc.halo.dn = l->getLayerLoc()->halo.dn;
   sLoc.halo.up = l->getLayerLoc()->halo.up;

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
#ifdef DEBUG_PRINT
            printf("sCube val = %5i:, kex = %5i:, k = %5i\n", (int) sCube->data[kex], kex, k);
#endif
        }
      }
   }

#ifdef DEBUG_PRINT
   // write out extended cube values
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
#endif

   // this is the function we're testing...
   for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
      l->mirrorInteriorToBorder(borderId, sCube, bCube);
   }

#ifdef DEBUG_PRINT
   // write out extended cube values
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
#endif

   // check values at mirror indices
   // uses a completely different algorithm than mirrorInteriorToBorder

   // northwest
   for (int ky = kFirst; ky < kFirst+nB; ky++) {
      int kymirror = kFirst - 1 - (ky - kFirst);
      for (int kx = kFirst; kx < kFirst+nB; kx++) {
         int kxmirror = kFirst - 1 - (kx - kFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:northwest mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   // north
   for (int ky = kFirst; ky < kFirst+nB; ky++) {
      int kymirror = kFirst - 1 - (ky - kFirst);
      for (int kx = kFirst; kx < kLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:north mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   // northeast
   for (int ky = kFirst; ky < kFirst+nB; ky++) {
      int kymirror = kFirst - 1 - (ky - kFirst);
      for (int kx = kLast - nB; kx < kLast; kx++) {
         int kxmirror = kLast - 1 + (kLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:northeast mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   // west
   for (int ky = kFirst; ky < kLast; ky++) {
      int kymirror = ky;
      for (int kx = kFirst; kx < kFirst + nB; kx++) {
         int kxmirror = kFirst - 1 - (kx - kFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:west mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }


   // east
   for (int ky = kFirst; ky < kLast; ky++) {
      int kymirror = ky;
      for (int kx = kLast - nB; kx < kLast; kx++) {
         int kxmirror = kLast - 1 + (kLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:east mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   // southwest
   for (int ky = kLast - nB; ky < kLast; ky++) {
      int kymirror = kLast - 1 + (kLast - ky);
      for (int kx = kFirst; kx < kFirst+nB; kx++) {
         int kxmirror = kFirst - 1 - (kx - kFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:southwest mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }

   // south
   for (int ky = kLast - nB; ky < kLast; ky++) {
      int kymirror = kLast - 1 + (kLast - ky);
      for (int kx = kFirst; kx < kLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:south mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
               exit(1);
            }
         }
      }
   }


   // southeast
   for (int ky = kLast - nB; ky < kLast; ky++) {
      int kymirror = kLast - 1 + (kLast - ky);
      for (int kx = kLast - nB; kx < kLast; kx++) {
         int kxmirror = kLast - 1 + (kLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex = ky * syex + kx * nf + kf;
            int k = (ky-kFirst) * sy + (kx-kFirst) * nf + kf;
            int kmirror = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if ( mirrorVal != k) {
               printf("ERROR:southeast mirror value at %i from %i = %i, should be %i\n", kmirror, kex, mirrorVal, k);
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
