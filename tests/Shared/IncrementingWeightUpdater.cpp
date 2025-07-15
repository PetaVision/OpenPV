/*
 * IncrementingWeightUpdater.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#include "IncrementingWeightUpdater.hpp"

namespace PV {

IncrementingWeightUpdater::IncrementingWeightUpdater(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

void IncrementingWeightUpdater::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HebbianUpdater::initialize(paramsIO, comm);
}

int IncrementingWeightUpdater::updateWeights(int arborId) {
   int nPatch         = mWeights->getPatchSizeOverall();
   int numDataPatches = mWeights->getNumDataPatches();
   for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
      float *Wdata  = mWeights->getDataFromDataIndex(arborId, patchIndex);
      float *dWdata = mDeltaWeights->getDataFromDataIndex(arborId, patchIndex);
      for (int k = 0; k < nPatch; k++) {
         float const dw = 1.0f;
         dWdata[k]      = dw;
         Wdata[k] += dw;
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
