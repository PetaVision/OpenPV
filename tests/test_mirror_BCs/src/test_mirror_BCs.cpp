/**
 * This file tests copying to boundary regions while applying mirror boundary conditions.
 * formerly called test_borders.cpp
 *
 */

#undef DEBUG_PRINT

#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "layers/HyPerLayer.hpp"

// const int numFeatures = 1;

int main(int argc, char *argv[]) {
   PV::PV_Init *initObj = new PV::PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);

   PV::HyPerCol *hc = new PV::HyPerCol(initObj);
   PV::HyPerLayer *layer =
         dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName("test_mirror_BCs_layer"));

   auto *layerGeometry = layer->getComponentByType<PV::LayerGeometry>();
   FatalIf(layerGeometry == nullptr, "%s does not have a LayerGeometry component.\n");
   int margin = 2;
   layerGeometry->requireMarginWidth(margin, 'x');
   layerGeometry->requireMarginWidth(margin, 'y');

   hc->processParams(hc->getPrintParamsFilename());

   auto *boundaryConditions = layer->getComponentByType<PV::BoundaryConditions>();
   FatalIf(boundaryConditions == nullptr, "%s does not have a BoundaryConditions component.\n");
   FatalIf(!boundaryConditions->getMirrorBCflag(), "%s has mirrorBCflag set to false.\n");

   std::vector<float> testBuffer(layer->getNumExtended(), 0.0f);
   PVLayerLoc const *loc = layer->getLayerLoc();

   // fill interior with non-extended index of each neuron
   // leave border values at zero to start with
   int kxFirst = loc->halo.lt;
   int kxLast  = loc->nx + loc->halo.lt;
   int kyFirst = loc->halo.up;
   int kyLast  = loc->ny + loc->halo.up;
   int nf      = loc->nf;
   int sy      = loc->nx * nf;
   int syex    = (loc->nx + loc->halo.lt + loc->halo.rt) * nf;
   for (int ky = kyFirst; ky < kyLast; ky++) {
      for (int kx = kxFirst; kx < kxLast; kx++) {
         for (int kf = 0; kf < loc->nf; kf++) {
            int kex         = ky * syex + kx * nf + kf;
            int kGlobal     = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            testBuffer[kex] = kGlobal;
         }
      }
   }

   // this is the function we're testing...
   boundaryConditions->applyBoundaryConditions(testBuffer.data(), loc);

   // check values at mirror indices
   // uses a completely different algorithm than mirrorInteriorToBorder

   // northwest
   for (int ky = kyFirst; ky < kyFirst + loc->halo.lt; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxFirst; kx < kxFirst + loc->halo.lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:northwest mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // north
   for (int ky = kyFirst; ky < kyFirst + loc->halo.up; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxFirst; kx < kxLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:north mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // northeast
   for (int ky = kyFirst; ky < kyFirst + loc->halo.up; ky++) {
      int kymirror = kyFirst - 1 - (ky - kyFirst);
      for (int kx = kxLast - loc->halo.rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kxFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:northeast mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // west
   for (int ky = kyFirst; ky < kyLast; ky++) {
      int kymirror = ky;
      for (int kx = kxFirst; kx < kxFirst + loc->halo.lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:west mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // east
   for (int ky = kyFirst; ky < kyLast; ky++) {
      int kymirror = ky;
      for (int kx = kxLast - loc->halo.rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:east mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // southwest
   for (int ky = kyLast - loc->halo.dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxFirst; kx < kxFirst + loc->halo.lt; kx++) {
         int kxmirror = kxFirst - 1 - (kx - kxFirst);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:southwest mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // south
   for (int ky = kyLast - loc->halo.dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxFirst; kx < kxLast; kx++) {
         int kxmirror = kx;
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:south mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   // southeast
   for (int ky = kyLast - loc->halo.dn; ky < kyLast; ky++) {
      int kymirror = kyLast - 1 + (kyLast - ky);
      for (int kx = kxLast - loc->halo.rt; kx < kxLast; kx++) {
         int kxmirror = kxLast - 1 + (kxLast - kx);
         for (int kf = 0; kf < nf; kf++) {
            int kex       = ky * syex + kx * nf + kf;
            int kGlobal   = (ky + loc->ky0 - kyFirst) * sy + (kx + loc->kx0 - kxFirst) * nf + kf;
            int kmirror   = kymirror * syex + kxmirror * nf + kf;
            int mirrorVal = testBuffer[kmirror];
            if (mirrorVal != kGlobal) {
               Fatal().printf(
                     "ERROR:southeast mirror value at %i from %i = %i, should be %i\n",
                     kmirror,
                     kex,
                     mirrorVal,
                     kGlobal);
            }
         }
      }
   }

   delete hc;
   delete initObj;

   return 0;
}
