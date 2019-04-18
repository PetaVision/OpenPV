/**
 * This file tests copying to boundary regions while applying mirror boundary conditions.
 * formerly called test_borders.cpp
 *
 */

#undef DEBUG_PRINT

#include "columns/PV_Init.hpp"
#include "io/io.hpp"
#include "layers/HyPerLayer.hpp"

// const int numFeatures = 1;

int main(int argc, char *argv[]) {
   PV::PV_Init *initObj = new PV::PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   PVLayerLoc sLoc, bLoc;
   PVLayerCube *sCube, *bCube;

   PV::HyPerCol *hc = new PV::HyPerCol(initObj);
   PV::HyPerLayer *l =
         dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName("test_mirror_BCs_layer"));

   int nf             = l->clayer->loc.nf;
   PVHalo const *halo = &l->clayer->loc.halo;
   int nS             = l->clayer->loc.nx; // 8;
   int syex           = (nS + halo->lt + halo->rt) * nf;
   int sy             = nS * nf;

   sLoc.nbatch   = 1;
   sLoc.nxGlobal = sLoc.nyGlobal = nS; // shouldn't be used
   sLoc.kx0 = sLoc.ky0 = 0; // shouldn't be used
   sLoc.nx = sLoc.ny = nS;
   sLoc.nf           = nf;
   sLoc.halo.lt      = halo->lt;
   sLoc.halo.rt      = halo->rt;
   sLoc.halo.dn      = halo->dn;
   sLoc.halo.up      = halo->up;

   bLoc = sLoc;

   sCube = PV::pvcube_new(&sLoc, (nS + halo->lt + halo->rt) * (nS + halo->dn + halo->up) * nf);
   bCube = sCube;

   // fill interior with non-extended index of each neuron
   // leave border values at zero to start with
   int kxFirst = halo->lt;
   int kxLast  = nS + halo->lt;
   int kyFirst = halo->up;
   int kyLast  = nS + halo->up;
   for (int ky = kyFirst; ky < kyLast; ky++) {
      for (int kx = kxFirst; kx < kxLast; kx++) {
         for (int kf = 0; kf < nf; kf++) {
            int kex          = ky * syex + kx * nf + kf;
            int k            = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            sCube->data[kex] = k;
#ifdef DEBUG_PRINT
            DebugLog().printf(
                  "sCube val = %5i:, kex = %5i:, k = %5i\n", (int)sCube->data[kex], kex, k);
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
            DebugLog().printf("%5i ", (int)sCube->data[kex]);
         }
         DebugLog().printf("\n");
      }
      DebugLog().printf("\n");
   }
#endif

   // this is the function we're testing...
   l->mirrorInteriorToBorder(sCube, bCube);

#ifdef DEBUG_PRINT
   // write out extended cube values
   for (int kf = 0; kf < nf; kf++) {
      for (int ky = 0; ky < ny; ky++) {
         for (int kx = 0; kx < nx; kx++) {
            int kex = ky * syex + kx * nf + kf;
            DebugLog().printf("%5i ", (int)sCube->data[kex]);
         }
         DebugLog().printf("\n");
      }
      DebugLog().printf("\n");
   }
#endif

   // check values at mirror indices
   // uses a completely different algorithm than mirrorInteriorToBorder

   // northwest
   for (int ky = kxFirst; ky < kyFirst + halo->lt; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxFirst; kx < kxFirst + halo->lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:northwest mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // north
   for (int ky = kyFirst; ky < kyFirst + halo->up; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxFirst; kx < kxLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:north mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // northeast
   for (int ky = kyFirst; ky < kyFirst + halo->up; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxLast - halo->rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kxFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:northeast mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // west
   for (int ky = kyFirst; ky < kyLast; ky++) {
      int kymirror = ky;
      for (int kx = kxFirst; kx < kxFirst + halo->lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:west mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // east
   for (int ky = kyFirst; ky < kyLast; ky++) {
      int kymirror = ky;
      for (int kx = kxLast - halo->rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kyFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:east mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // southwest
   for (int ky = kyLast - halo->dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxFirst; kx < kxFirst + halo->lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:southwest mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // south
   for (int ky = kyLast - halo->dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxFirst; kx < kxLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:south mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   // southeast
   for (int ky = kyLast - halo->dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxLast - halo->rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int k         = (ky - kyFirst) * sy + (kx - kyFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = bCube->data[kmirror];
            if (mirrorVal != k) {
               Fatal().printf(
                     "ERROR:southeast mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     k);
            }
         }
      }
   }

   PV::pvcube_delete(sCube);
   sCube = bCube = NULL;

   delete hc;
   delete initObj;

   return 0;
}
