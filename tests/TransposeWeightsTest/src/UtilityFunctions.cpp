#include "UtilityFunctions.hpp"
#include <components/Weights.cpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/TransposeWeights.hpp>
#include <utils/conversions.h>

int calcStride(int pre, std::string const &preDesc, int post, std::string const &postDesc) {
   int stride;
   if (pre > post) {
      stride = pre / post;
      FatalIf(
            stride * post != pre,
            "calcStride called with %s=%d greater than %s=%d, but not an even multiple.\n",
            preDesc.c_str(),
            pre,
            postDesc.c_str(),
            post);
      FatalIf(
            stride % 2 != 0,
            "calcStride called with %s=%d greater than %s=%d, for a stride of %d, "
            "but the stride must be even.\n",
            preDesc.c_str(),
            pre,
            postDesc.c_str(),
            post,
            stride);
   }
   else {
      stride = 1;
   }
   return stride;
}

void checkMPICompatibility(PVLayerLoc const &loc, PV::Communicator *comm) {
   FatalIf(
         loc.nbatch * comm->numCommBatches() != loc.nbatchGlobal,
         "Number of processes in the batch dimension is %d, but it must be a "
         "divisor of batch width %d\n",
         comm->numCommBatches(),
         loc.nbatchGlobal);
   FatalIf(
         loc.nx * comm->numCommColumns() != loc.nxGlobal,
         "Number of processes in the x-dimension is %d, but it must be a "
         "divisor of layer width %d\n",
         comm->numCommColumns(),
         loc.nxGlobal);
   FatalIf(
         loc.ny * comm->numCommRows() != loc.nyGlobal,
         "Number of processes in the y-dimension is %d, but it must be a "
         "divisor of layer height %d\n",
         comm->numCommRows(),
         loc.nyGlobal);
}

PV::Weights createOriginalWeights(
      bool sharedFlag,
      int nxPre,
      int nyPre,
      int nfPre,
      int nxPost,
      int nyPost,
      int nfPost,
      int patchSizeXPre,
      int patchSizeYPre,
      PV::Communicator *comm) {
   int const xStride  = calcStride(nxPre, std::string("nxPre"), nxPost, std::string("nxPost"));
   int const xTStride = calcStride(nxPost, std::string("nxPost"), nxPre, std::string("nxPre"));
   int const yStride  = calcStride(nyPre, std::string("nyPre"), nyPost, std::string("nyPost"));
   int const yTStride = calcStride(nyPost, std::string("nyPost"), nyPre, std::string("nyPre"));

   int const patchSizeXPost = patchSizeXPre * nxPre / nxPost;
   int const marginXPre     = (patchSizeXPost - xStride) / 2;
   pvAssert(2 * marginXPre == patchSizeXPost - xStride);

   int const patchSizeYPost = patchSizeYPre * nxPre / nxPost;
   int const marginYPre     = (patchSizeYPost - xStride) / 2;
   pvAssert(2 * marginYPre == patchSizeYPost - yStride);

   int const marginXPost = (patchSizeXPre - xTStride) / 2;
   pvAssert(2 * marginXPost == patchSizeXPre - xTStride);

   int const marginYPost = (patchSizeYPre - yTStride) / 2;
   pvAssert(2 * marginYPost == patchSizeYPre - yTStride);

   PVLayerLoc preLoc;
   preLoc.nbatchGlobal = 1;
   preLoc.nbatch       = preLoc.nbatchGlobal / comm->numCommBatches();
   preLoc.kb0          = 0;
   preLoc.nxGlobal     = nxPre;
   preLoc.nx           = preLoc.nxGlobal / comm->numCommColumns();
   preLoc.kx0          = preLoc.nx * comm->commColumn();
   preLoc.nyGlobal     = nyPre;
   preLoc.ny           = preLoc.nyGlobal / comm->numCommRows();
   preLoc.ky0          = preLoc.ny * comm->commRow();
   preLoc.nf           = nfPre;
   preLoc.halo.lt      = marginXPre;
   preLoc.halo.rt      = marginXPre;
   preLoc.halo.dn      = marginYPre;
   preLoc.halo.up      = marginYPre;
   checkMPICompatibility(preLoc, comm);

   PVLayerLoc postLoc;
   postLoc.nbatchGlobal = 1;
   postLoc.nbatch       = postLoc.nbatchGlobal / comm->numCommBatches();
   postLoc.kb0          = postLoc.nbatch * comm->commBatch();
   postLoc.nxGlobal     = nxPost;
   postLoc.nx           = postLoc.nxGlobal / comm->numCommColumns();
   postLoc.kx0          = postLoc.nx * comm->commColumn();
   postLoc.nyGlobal     = nyPost;
   postLoc.ny           = postLoc.nyGlobal / comm->numCommRows();
   postLoc.ky0          = postLoc.ny * comm->commRow();
   postLoc.nf           = nfPost;
   postLoc.halo.lt      = marginXPost;
   postLoc.halo.rt      = marginXPost;
   postLoc.halo.dn      = marginYPost;
   postLoc.halo.up      = marginYPost;
   checkMPICompatibility(postLoc, comm);

   std::string originalWeightsName("Original");
   int const patchSizeFPre = nfPost;
   int const numArbors     = 1;
   double const timestamp  = 0.0;
   PV::Weights originalWeights(
         originalWeightsName,
         patchSizeXPre,
         patchSizeYPre,
         patchSizeFPre,
         &preLoc,
         &postLoc,
         numArbors,
         sharedFlag,
         timestamp);

   int const numKernelsPre = originalWeights.getGeometry()->getNumKernels();
   FatalIf(
         numKernelsPre != xStride * yStride * preLoc.nf,
         "originalWeights should have numKernelsPre=%d; value was %d.\n",
         preLoc.nf,
         numKernelsPre);

   int const numPatchItemsPre = originalWeights.getPatchSizeOverall();
   FatalIf(
         numPatchItemsPre != patchSizeXPre * patchSizeYPre * patchSizeFPre,
         "originalWeights should have numPatchItemsPre=%d; value was %d.\n",
         patchSizeXPre * patchSizeYPre * patchSizeFPre,
         numPatchItemsPre);

   originalWeights.allocateDataStructures();

   if (sharedFlag) {
      // Initialize the values of originalWeights to the values 1 through
      // numKernelsPre*numPatchItemsPre. We don't zero-index because a zero
      // value arises accidentally too easily.
      for (int patchIndex = 0; patchIndex < numKernelsPre; patchIndex++) {
         float *patchData = originalWeights.getDataFromDataIndex(0 /*arbor index*/, patchIndex);
         for (int itemInPatch = 0; itemInPatch < numPatchItemsPre; itemInPatch++) {
            patchData[itemInPatch] = (float)(1 + itemInPatch + patchIndex * numPatchItemsPre);
         }
      }
   }
   else {
      // Initialize the values of originalWeights.
      // Use the global restricted presynaptic neuron index and patch item, so that each weight
      // value is unique.
      // Neurons in the global border region have their weights initialized to zero, because these
      // weights don't show up in the transpose; it is only under this constraint that
      // transpose of transpose equals original.
      // Neurons in interior border regions do get nonzero values.
      // By the same token, we only initialize values in the shrunken patches.
      int const numGlobalRestricted = preLoc.nxGlobal * preLoc.nyGlobal * preLoc.nf;
      int const numItemsInPatch     = originalWeights.getPatchSizeOverall();
      for (int globalRestricted = 0; globalRestricted < numGlobalRestricted; globalRestricted++) {
         int const xGlobal = kxPos(globalRestricted, preLoc.nxGlobal, preLoc.nyGlobal, preLoc.nf);
         int const xLocalExtended = xGlobal - preLoc.kx0 + preLoc.halo.lt;
         if (xLocalExtended < 0 or xLocalExtended >= preLoc.nx + preLoc.halo.lt + preLoc.halo.rt) {
            continue;
         }

         int const yGlobal = kyPos(globalRestricted, preLoc.nxGlobal, preLoc.nyGlobal, preLoc.nf);
         int const yLocalExtended = yGlobal - preLoc.ky0 + preLoc.halo.up;
         if (yLocalExtended < 0 or yLocalExtended >= preLoc.ny + preLoc.halo.dn + preLoc.halo.up) {
            continue;
         }

         int const fLocalExtended =
               featureIndex(globalRestricted, preLoc.nxGlobal, preLoc.nyGlobal, preLoc.nf);
         int const localExtended = kIndex(
               xLocalExtended,
               yLocalExtended,
               fLocalExtended,
               preLoc.nx + preLoc.halo.lt + preLoc.halo.rt,
               preLoc.ny + preLoc.halo.dn + preLoc.halo.up,
               preLoc.nf);
         PV::Patch const &patch = originalWeights.getPatch(localExtended);
         for (int itemInPatch = 0; itemInPatch < numItemsInPatch; itemInPatch++) {
            float value;
            // Are we inside a shrunken patch?
            int const xStart = kxPos(patch.offset, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            int const yStart = kyPos(patch.offset, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            int const x      = kxPos(itemInPatch, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            int const y      = kyPos(itemInPatch, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            if (x >= xStart and x < xStart + patch.nx and y >= yStart and y < yStart + patch.ny) {
               value = (float)(1 + itemInPatch + globalRestricted * numItemsInPatch);
            }
            else {
               value = 0.0f;
            }
            float *patchData = originalWeights.getDataFromDataIndex(0 /*arbor*/, localExtended);
            patchData[itemInPatch] = value;
         }
      }
   }

   return originalWeights;
}

int checkTransposeOfTranspose(
      std::string const &testName,
      PV::Weights &originalWeights,
      PV::Weights &transposeWeights,
      PV::Communicator *comm) {
   int status = PV_SUCCESS;

   PV::Weights transposeOfTranspose(std::string("transpose of transpose"));
   transposeOfTranspose.initialize(
         originalWeights.getPatchSizeX(),
         originalWeights.getPatchSizeY(),
         originalWeights.getPatchSizeF(),
         &originalWeights.getGeometry()->getPreLoc(),
         &originalWeights.getGeometry()->getPostLoc(),
         originalWeights.getNumArbors(),
         originalWeights.getSharedFlag(),
         originalWeights.getTimestamp());
   transposeOfTranspose.allocateDataStructures();
   PV::TransposeWeights::transpose(&transposeWeights, &transposeOfTranspose, comm);

   int const numDataPatchesPre = originalWeights.getNumDataPatches();
   if (transposeOfTranspose.getNumDataPatches() != numDataPatchesPre) {
      ErrorLog().printf(
            "In %s, transpose of transpose has %d data patches instead "
            "of the expected %d.\n",
            testName.c_str(),
            transposeOfTranspose.getNumDataPatches(),
            numDataPatchesPre);
      status = PV_FAILURE;
   }

   int const numPatchItemsPre = originalWeights.getPatchSizeOverall();
   if (transposeOfTranspose.getPatchSizeOverall() != numPatchItemsPre) {
      ErrorLog().printf(
            "In %s, transpose of transpose has an overall patch size of %d "
            "instead of the expected %d.\n",
            testName.c_str(),
            transposeOfTranspose.getPatchSizeOverall(),
            numPatchItemsPre);
      status = PV_FAILURE;
   }

   for (int k = 0; k < numDataPatchesPre; k++) {
      for (int i = 0; i < numPatchItemsPre; i++) {
         float observed = transposeOfTranspose.getDataFromDataIndex(0, k)[i];
         float expected = originalWeights.getDataFromDataIndex(0, k)[i];
         if (expected != observed) {
            ErrorLog().printf(
                  "In %s, data patch %d, patch item %d, "
                  "transposeOfTranspose has value %d but original has value %d.\n",
                  testName.c_str(),
                  k,
                  i,
                  (int)observed,
                  (int)expected);
            status = PV_FAILURE;
         }
      }
   }

   return status;
}
