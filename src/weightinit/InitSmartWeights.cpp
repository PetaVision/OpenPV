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

void InitSmartWeights::calcWeights(long patchIndex, int arborId) {
   float *dataStart = mWeights->getDataFromDataIndex(arborId, patchIndex);
   smartWeights(dataStart, patchIndex);
}

void InitSmartWeights::smartWeights(float *dataStart, long k) {
   long const nfp = (long)mWeights->getPatchSizeF();
   long const nyp = (long)mWeights->getPatchSizeY();
   long const nxp = (long)mWeights->getPatchSizeX();

   long const sxp = (long)mWeights->getGeometry()->getPatchStrideX();
   long const syp = (long)mWeights->getGeometry()->getPatchStrideY();
   long const sfp = (long)mWeights->getGeometry()->getPatchStrideF();

   // loop over all post-synaptic cells in patch
   for (long y = 0; y < nyp; y++) {
      for (long x = 0; x < nxp; x++) {
         for (long f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = (float)dataIndexToUnitCellIndex(k);
         }
      }
   }
}

} /* namespace PV */
