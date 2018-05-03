#include "TestNonshared.hpp"
#include "UtilityFunctions.hpp"
#include <components/Weights.hpp>
#include <components/WeightsPair.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/TransposeWeights.hpp>
#include <utils/conversions.h>

#include <cmath>

int TestNonshared(
      std::string const &testName,
      int nxPre,
      int nyPre,
      int nfPre,
      int nxPost,
      int nyPost,
      int nfPost,
      int patchSizeXPre,
      int patchSizeYPre,
      PV::Communicator *comm) {
   int status = PV_SUCCESS;

   // For one-to-one connections, stride and transpose-stride are both one.
   // For many-to-one connections, stride > 1 and transpose-stride == 1.
   // For one-to-many connections, stride == 1 and transpose-stride > 1.
   // The formulas in this function can seem more complicated than they really are, because they
   // combine the one-to-one, many-to-one, and one-to-many cases, in order to avoid numerous
   // if/if else/else constructions. But in all cases at least one of xStride and xTStride is
   // equal to one, and at least one of yStride and yTStride is equal to one.
   int const xStride  = calcStride(nxPre, std::string("nxPre"), nxPost, std::string("nxPost"));
   int const xTStride = calcStride(nxPost, std::string("nxPost"), nxPre, std::string("nxPre"));
   int const yStride  = calcStride(nyPre, std::string("nyPre"), nyPost, std::string("nyPost"));
   int const yTStride = calcStride(nyPost, std::string("nyPost"), nyPre, std::string("nyPre"));

   FatalIf(
         xStride == 1 and xTStride == 1 and patchSizeXPre % 2 != 1,
         "%s has nxPre=%d and nxPost=%d equal, but patchSizeXPre=%d is not odd.\n",
         testName.c_str(),
         nxPre,
         nxPost,
         patchSizeXPre);

   FatalIf(
         yStride == 1 and yTStride == 1 and patchSizeYPre % 2 != 1,
         "%s has nyPre=%d and nyPost=%d equal, but patchSizeYPre=%d is not odd.\n",
         testName.c_str(),
         nyPre,
         nyPost,
         patchSizeYPre);

   FatalIf(
         xTStride > 1 and patchSizeXPre % xTStride != 0,
         "%s has nxPre=%d and nxPost=%d for a transpose-stride of %d, "
         "but patchSizeXPre %d is not a multiple of the transpose-stride.\n",
         testName.c_str(),
         nxPre,
         nxPost,
         xTStride,
         patchSizeXPre,
         xTStride);
   int const patchSizeXPost = xStride * patchSizeXPre / xTStride;

   FatalIf(
         yTStride > 1 and patchSizeYPre % yTStride != 0,
         "%s has nyPre=%d and nyPost=%d for a transpose-stride of %d, "
         "but patchSizeYPre %d is not a multiple of the transpose-stride.\n",
         testName.c_str(),
         nyPre,
         nyPost,
         yTStride,
         patchSizeXPre,
         yTStride);
   int const patchSizeYPost = yStride * patchSizeYPre / yTStride;

   int const patchSizeFPost = nfPre;

   bool const nonShared        = false;
   PV::Weights originalWeights = createOriginalWeights(
         nonShared,
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeXPre,
         patchSizeYPre,
         comm);

   int const patchSizeFPre    = originalWeights.getPatchSizeF();
   int const numPatchItemsPre = originalWeights.getPatchSizeOverall();
   int const numKernelsPre    = originalWeights.getGeometry()->getNumKernels();

   std::string transposeWeightsName("Transpose");
   PV::Weights transposeWeights(
         transposeWeightsName,
         PV::PatchSize::calcPostPatchSize(patchSizeXPre, nxPre, nxPost),
         PV::PatchSize::calcPostPatchSize(patchSizeYPre, nyPre, nyPost),
         nfPre,
         &originalWeights.getGeometry()->getPostLoc(),
         &originalWeights.getGeometry()->getPreLoc(),
         1 /*numArbors*/,
         nonShared,
         0.0 /*timestamp*/);

   int const numPatchesPost = transposeWeights.getGeometry()->getNumPatches();

   int const numPatchItemsPost = transposeWeights.getPatchSizeOverall();
   FatalIf(
         transposeWeights.getPatchSizeX() != patchSizeXPost,
         "%s has transposeWeights PatchSizeX of %d, when it should be %d.\n",
         testName.c_str(),
         transposeWeights.getPatchSizeX(),
         patchSizeXPost);
   FatalIf(
         transposeWeights.getPatchSizeY() != patchSizeYPost,
         "%s has transposeWeights PatchSizeY of %d, when it should be %d.\n",
         testName.c_str(),
         transposeWeights.getPatchSizeY(),
         patchSizeYPost);
   FatalIf(
         transposeWeights.getPatchSizeF() != patchSizeFPost,
         "%s has transposeWeights PatchSizeF of %d, when it should be %d.\n",
         testName.c_str(),
         transposeWeights.getPatchSizeF(),
         patchSizeFPost);
   FatalIf(
         numPatchItemsPost != patchSizeXPost * patchSizeYPost * patchSizeFPost,
         "%s has transposeWeights overall patch size of %d, when it should be %d.\n",
         testName.c_str(),
         numPatchItemsPost,
         patchSizeXPost * patchSizeYPost * patchSizeFPost);

   transposeWeights.allocateDataStructures();

   PV::TransposeWeights::transpose(&originalWeights, &transposeWeights, comm);

   // Check the values
   int const numPatchesXPost = transposeWeights.getNumDataPatchesX();
   int const numPatchesYPost = transposeWeights.getNumDataPatchesY();
   int const numPatchesFPost = transposeWeights.getNumDataPatchesF();

   auto geometryPost = transposeWeights.getGeometry();

   // preLoc and postLoc are from the original weights' point of view,
   // so postLoc is for the transposeWeight's presynaptic layer and vice versa.
   PVLayerLoc const &postLoc = geometryPost->getPreLoc();
   PVLayerLoc const &preLoc  = geometryPost->getPostLoc();
   for (int patchIndexPost = 0; patchIndexPost < numPatchesPost; patchIndexPost++) {
      for (int itemInPatchPost = 0; itemInPatchPost < numPatchItemsPost; itemInPatchPost++) {
         PV::Patch const &patchPost = transposeWeights.getPatch(patchIndexPost);

         int const patchOffsetXPost =
               kxPos(patchPost.offset, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const itemInPatchXPost =
               kxPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const fromOffsetXPost   = itemInPatchXPost - patchOffsetXPost;
         bool const xInShrunkenPatch = fromOffsetXPost >= 0 and fromOffsetXPost < patchPost.nx;

         int const patchOffsetYPost =
               kyPos(patchPost.offset, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const itemInPatchYPost =
               kyPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const fromOffsetYPost   = itemInPatchYPost - patchOffsetYPost;
         bool const yInShrunkenPatch = fromOffsetYPost >= 0 and fromOffsetYPost < patchPost.ny;

         int const kxLocalExtPost =
               kxPos(patchIndexPost, numPatchesXPost, numPatchesYPost, numPatchesFPost);
         int const kxGlobalResPost = kxLocalExtPost + postLoc.kx0 - postLoc.halo.lt;
         bool const xInBoundsPost  = kxGlobalResPost >= 0 and kxGlobalResPost < postLoc.nxGlobal;

         int const kyLocalExtPost =
               kyPos(patchIndexPost, numPatchesXPost, numPatchesYPost, numPatchesFPost);
         int const kyGlobalResPost = kyLocalExtPost + postLoc.ky0 - postLoc.halo.dn;
         bool const yInBoundsPost  = kyGlobalResPost >= 0 and kyGlobalResPost < postLoc.nyGlobal;

         int const kfLocalExtPost =
               featureIndex(patchIndexPost, numPatchesXPost, numPatchesYPost, numPatchesFPost);
         int const kfGlobalResPost = kfLocalExtPost;

         float correctValue;
         if (xInShrunkenPatch and yInShrunkenPatch and xInBoundsPost and yInBoundsPost) {
            int const itemInPatchXPost =
                  kxPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
            int const itemInPatchYPost =
                  kyPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
            int const itemInPatchFPost =
                  featureIndex(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);

            int const patchStartXInPre = PV::PatchGeometry::calcPatchStartInPost(
                  kxGlobalResPost, patchSizeXPost, postLoc.nxGlobal, preLoc.nxGlobal);
            int const kxGlobalResPre = patchStartXInPre + itemInPatchXPost;
            bool const xInBoundsPre  = kxGlobalResPre >= 0 and kxGlobalResPre < preLoc.nxGlobal;

            int const patchStartYInPre = PV::PatchGeometry::calcPatchStartInPost(
                  kyGlobalResPost, patchSizeYPost, postLoc.nyGlobal, preLoc.nyGlobal);
            int const kyGlobalResPre = patchStartYInPre + itemInPatchYPost;
            bool const yInBoundsPre  = kyGlobalResPre >= 0 and kyGlobalResPre < preLoc.nyGlobal;

            if (xInBoundsPre and yInBoundsPre) {
               int const kfGlobalResPre = itemInPatchFPost;

               int const kernelXPost = kxGlobalResPost % geometryPost->getNumKernelsX();
               int const kernelYPost = kyGlobalResPost % geometryPost->getNumKernelsY();
               int const kernelFPost = kfGlobalResPost;
               int const kernelPost  = kIndex(
                     kernelXPost,
                     kernelYPost,
                     kernelFPost,
                     geometryPost->getNumKernelsX(),
                     geometryPost->getNumKernelsY(),
                     geometryPost->getNumKernelsF());
               int const itemIndexPre =
                     geometryPost->getTransposeItemIndex(kernelPost, itemInPatchPost);

               int const kGlobalResPre = kIndex(
                     kxGlobalResPre,
                     kyGlobalResPre,
                     kfGlobalResPre,
                     preLoc.nxGlobal,
                     preLoc.nyGlobal,
                     preLoc.nf);

               int const patchSizePre = originalWeights.getPatchSizeOverall();
               correctValue           = (float)(1 + itemIndexPre + kGlobalResPre * patchSizePre);
            }
            else {
               correctValue = 0.0f;
            }
         }
         else {
            correctValue = 0.0f;
         }
         float const observedValue = transposeWeights.getDataFromDataIndex(
               0 /*arbor index*/, patchIndexPost)[itemInPatchPost];
         if (observedValue != correctValue) {
            ErrorLog().printf(
                  "%s, rank %d, patch index %d, patch item %d has transposeWeights value %d, "
                  "when it should be %d.\n",
                  testName.c_str(),
                  comm->globalCommRank(),
                  patchIndexPost,
                  itemInPatchPost,
                  (int)std::nearbyint(observedValue),
                  (int)std::nearbyint(correctValue));
            status = PV_FAILURE;
         }
      }
   }

   // As a second check, the transpose of the transpose should be the same as the original weights.
   if (checkTransposeOfTranspose(testName, originalWeights, transposeWeights, comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   return status;
}
