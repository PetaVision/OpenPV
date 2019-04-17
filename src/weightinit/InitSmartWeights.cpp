/*
 * InitSmartWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitSmartWeights.hpp"

namespace PV {

InitSmartWeights::InitSmartWeights(char const *name, PVParams *params, Communicator const *comm)
      : InitWeights() {
   InitSmartWeights::initialize(name, params, comm);
}

InitSmartWeights::InitSmartWeights() {}

InitSmartWeights::~InitSmartWeights() {}

void InitSmartWeights::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InitWeights::initialize(name, params, comm);
}

void InitSmartWeights::calcWeights(int patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   smartWeights(dataStart, patchIndex);
}

void InitSmartWeights::smartWeights(float *dataStart, int k) {
   int const nfp = mWeights->getPatchSizeF();
   int const nyp = mWeights->getPatchSizeY();
   int const nxp = mWeights->getPatchSizeX();

   int const sxp = mWeights->getGeometry()->getPatchStrideX();
   int const syp = mWeights->getGeometry()->getPatchStrideY();
   int const sfp = mWeights->getGeometry()->getPatchStrideF();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = dataIndexToUnitCellIndex(k);
         }
      }
   }
}

} /* namespace PV */
