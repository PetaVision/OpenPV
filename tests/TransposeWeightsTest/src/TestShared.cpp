#include "TestShared.hpp"
#include "UtilityFunctions.hpp"
#include <components/Weights.hpp>
#include <components/WeightsPair.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/TransposeWeights.hpp>
#include <utils/conversions.h>

int TestShared(
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
         testName.c_str(),
         "%s has nyPre=%d and nyPost=%d for a transpose-stride of %d, "
         "but patchSizeYPre %d is not a multiple of the transpose-stride.\n",
         nyPre,
         nyPost,
         yTStride,
         patchSizeXPre,
         yTStride);
   int const patchSizeYPost = yStride * patchSizeYPre / yTStride;

   int const patchSizeFPost = nfPre;

   bool const shared           = true;
   PV::Weights originalWeights = createOriginalWeights(
         shared, nxPre, nyPre, nfPre, nxPost, nyPost, nfPost, patchSizeXPre, patchSizeYPre, comm);

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
         shared,
         0.0 /*timestamp*/);

   int const numKernelsPost = transposeWeights.getGeometry()->getNumKernels();
   FatalIf(
         numKernelsPost != xTStride * yTStride * nfPost,
         "%s expected numKernelsPost=%d; value was %d.\n",
         testName.c_str(),
         xTStride * yTStride * nfPost,
         numKernelsPost);

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
         "%s has overall patch size of %d, when it should be %d.\n",
         testName.c_str(),
         numPatchItemsPost,
         patchSizeXPost * patchSizeYPost * patchSizeFPost);

   transposeWeights.allocateDataStructures();

   PV::TransposeWeights::transpose(&originalWeights, &transposeWeights, comm);

   for (int kPost = 0; kPost < numKernelsPost; kPost++) {
      int const kxPost = kxPos(kPost, xTStride, yTStride, nfPost);
      int const kyPost = kyPos(kPost, xTStride, yTStride, nfPost);
      int const kfPost = featureIndex(kPost, xStride, yStride, nfPost);
      for (int iPost = 0; iPost < numPatchItemsPost; iPost++) {
         int const xPost = kxPos(iPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const yPost = kyPos(iPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const fPost = featureIndex(iPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);

         int const kxPre = xPost % xStride;
         int const kyPre = yPost % yStride;
         int const kfPre = fPost;
         int const kPre  = kIndex(kxPre, kyPre, kfPre, xStride, yStride, nfPre);

         int const xPre = xTStride * (patchSizeXPost - 1 - (xPost - kxPre)) / xStride + kxPost;
         int const yPre = yTStride * (patchSizeYPost - 1 - (yPost - kyPre)) / yStride + kyPost;
         int const fPre = kfPost;
         int const iPre = kIndex(xPre, yPre, fPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);

         float const postWeight = transposeWeights.getDataFromDataIndex(0, kPost)[iPost];
         float const preWeight  = originalWeights.getDataFromDataIndex(0, kPre)[iPre];
         if (postWeight != preWeight) {
            ErrorLog().printf(
                  "%s transposeWeights kernel %d has the value %d "
                  "at x=%d, y=%d, f=%d, instead of the expected %d.\n",
                  kPost,
                  (int)postWeight,
                  xPost,
                  yPost,
                  fPost,
                  (int)preWeight);
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
