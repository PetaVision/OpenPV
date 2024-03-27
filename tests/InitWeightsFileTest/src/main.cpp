/*
 * main function for InitWeightsFileTest
 *
 */

#include <columns/ComponentBasedObject.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <components/ConnectionData.hpp>
#include <components/PatchSize.hpp>
#include <components/WeightsPair.hpp>

#include <cmath>

using namespace PV;

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   HyPerCol *hc = new HyPerCol(&pv_initObj);
   if (hc == nullptr) {
      return EXIT_FAILURE;
   }
   hc->allocateColumn();
   auto *conn      = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("InputToOutput"));
   HyPerLayer *pre = conn->getComponentByType<ConnectionData>()->getPre();
   PVLayerLoc const preLoc = *pre->getLayerLoc();
   int const numExtended   = pre->getNumExtended();
   auto *preWeights        = conn->getComponentByType<WeightsPair>()->getPreWeights();
   int const numPatches    = preWeights->getNumDataPatches();
   FatalIf(
         numPatches != numExtended,
         "Presynaptic numExtended %d != number of data patches %d\n",
         numExtended,
         numPatches);

   auto *patchSize = conn->getComponentByType<PatchSize>();
   int const nxp   = patchSize->getPatchSizeX();
   int const nyp   = patchSize->getPatchSizeY();
   int const nfp   = patchSize->getPatchSizeF();

   float const numItemsInPatch = (float)(nxp * nyp * nfp);

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf    = preLoc.nf;

   int status = PV_SUCCESS;
   for (int index = 0; index < numExtended; index++) {
      int xExtIndex = kxPos(index, nxExt, nyExt, nf) + preLoc.kx0;
      int yExtIndex = kyPos(index, nxExt, nyExt, nf) + preLoc.ky0;
      int fIndex = featureIndex(index, nxExt, nyExt, nf);

      int nxExtGlobal = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
      int nyExtGlobal = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
      int globalIndex = kIndex(xExtIndex, yExtIndex, fIndex, nxExtGlobal, nyExtGlobal, nf);

      float const *weights = preWeights->getDataFromDataIndex(0 /*arbor*/, index);

      // only need to check in shrunken patch region.
      Patch const &patch = preWeights->getPatch(index);
      int xStart         = kxPos(patch.offset, nxp, nyp, nfp);
      int yStart         = kyPos(patch.offset, nxp, nyp, nfp);
      for (int y = yStart; y < yStart + patch.ny; y++) {
         for (int x = xStart; x < xStart + patch.nx; x++) {
            for (int f = 0; f < nf; f++) {
               int k                    = kIndex(x, y, f, nxp, nyp, nfp);
               float w                  = weights[k];
               float indexObservedFloat = std::floor(w);
               int indexObserved        = (int)indexObservedFloat;
               float kObservedFloat     = (w - indexObservedFloat) * numItemsInPatch;
               int kObserved            = (int)std::nearbyint(kObservedFloat);
               if (kObserved != k or indexObserved != globalIndex) {
                  ErrorLog().printf(
                        "Rank %d, Patch %d (global index %d), (x,y,z)=(%d,%d,%d): "
                        "expected %f; observed %f\n",
                        pv_initObj.getCommunicator()->globalCommRank(),
                        index,
                        globalIndex,
                        x,
                        y,
                        f,
                        (double)globalIndex + (double)k / (double)numItemsInPatch,
                        (double)w);
                  status = PV_FAILURE;
               }
            }
         }
      }
   }

   delete hc;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
