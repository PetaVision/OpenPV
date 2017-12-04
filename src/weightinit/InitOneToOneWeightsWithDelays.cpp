/*
 * InitOneToOneWeightsWithDelays.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#include "InitOneToOneWeightsWithDelays.hpp"

namespace PV {

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

InitOneToOneWeightsWithDelays::InitOneToOneWeightsWithDelays() {}

InitOneToOneWeightsWithDelays::~InitOneToOneWeightsWithDelays() {}

int InitOneToOneWeightsWithDelays::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

void InitOneToOneWeightsWithDelays::calcWeights(int patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   createOneToOneConnectionWithDelays(dataStart, patchIndex, mWeightInit, arborId);
}

void InitOneToOneWeightsWithDelays::createOneToOneConnectionWithDelays(
      float *dataStart,
      int dataPatchIndex,
      float iWeight,
      int arborId) {

   const int nArbors = mWeights->getNumArbors();
   int unitCellIndex = dataIndexToUnitCellIndex(dataPatchIndex);

   int const nfp = mWeights->getPatchSizeF();
   int const nyp = mWeights->getPatchSizeY();
   int const nxp = mWeights->getPatchSizeX();

   int const sxp = mWeights->getGeometry()->getPatchStrideX();
   int const syp = mWeights->getGeometry()->getPatchStrideY();
   int const sfp = mWeights->getGeometry()->getPatchStrideF();

   // clear all weights in patch
   memset(dataStart, 0, nxp * nyp * nfp);
   // then set the center point of the patch for each feature
   int x = (int)(nxp / 2);
   int y = (int)(nyp / 2);
   for (int f = 0; f < nfp; f++) {
      dataStart[x * sxp + y * syp + f * sfp] = f == nArbors * unitCellIndex + arborId ? iWeight : 0;
   }
}

} /* namespace PV */
