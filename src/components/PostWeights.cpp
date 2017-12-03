/*
 * PostWeights.cpp
 *
 *  Created on: Sep 1, 2017
 *      Author: Pete Schultz
 */

#include "PostWeights.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

PostWeights::PostWeights(std::string const &name) { setName(name); }

PostWeights::PostWeights(std::string const &name, Weights *preWeights) {
   setName(name);
   initializePostWeights(preWeights);
}

void PostWeights::initializePostWeights(Weights *preWeights) {
   auto preGeometry          = preWeights->getGeometry();
   PVLayerLoc const &preLoc  = preGeometry->getPreLoc();
   PVLayerLoc const &postLoc = preGeometry->getPostLoc();

   int const postPatchSizeX = calcPostPatchSize(preWeights->getPatchSizeX(), preLoc.nx, postLoc.nx);
   int const postPatchSizeY = calcPostPatchSize(preWeights->getPatchSizeX(), preLoc.ny, postLoc.ny);
   int const postPatchSizeF = preLoc.nf;

   Weights::initialize(
         postPatchSizeX,
         postPatchSizeY,
         postPatchSizeF,
         &postLoc,
         &preLoc,
         preWeights->getNumArbors(),
         preWeights->getSharedFlag(),
         preWeights->getTimestamp());
}

int PostWeights::calcPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost) {
   if (numNeuronsPre == numNeuronsPost) {
      return prePatchSize;
   }
   else if (numNeuronsPre > numNeuronsPost) {
      return manyToOnePostPatchSize(prePatchSize, numNeuronsPre, numNeuronsPost);
   }
   else {
      return oneToManyPostPatchSize(prePatchSize, numNeuronsPre, numNeuronsPost);
   }
}

int PostWeights::manyToOnePostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost) {
   pvAssert(numNeuronsPre > numNeuronsPost);
   std::div_t scaleDivision = div(numNeuronsPre, numNeuronsPost);
   FatalIf(
         scaleDivision.rem != 0,
         "calcPostPatchSize called with numNeuronsPre (%d) greater than numNeuronsPost (%d), "
         "but not an integer multiple.\n",
         numNeuronsPre,
         numNeuronsPost);
   return prePatchSize * scaleDivision.quot;
}

int PostWeights::oneToManyPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost) {
   pvAssert(numNeuronsPre < numNeuronsPost);
   std::div_t const scaleDivision = div(numNeuronsPost, numNeuronsPre);
   FatalIf(
         scaleDivision.rem != 0,
         "calcPostPatchSize called with numNeuronsPost (%d) greater than numNeuronsPre (%d), "
         "but not an integer multiple.\n",
         numNeuronsPost,
         numNeuronsPre);
   int const scaleFactor         = scaleDivision.quot;
   std::div_t const newPatchSize = div(prePatchSize, scaleFactor);
   FatalIf(
         newPatchSize.rem != 0,
         "calcPostPatchSize called with scale factor of numNeuronsPost/numNeuronsPre = %d, "
         "but prePatchSize (%d) is not an integer multiple of the scale factor.\n",
         scaleFactor,
         prePatchSize);
   return prePatchSize / scaleFactor;
}

} // end namespace PV
