/*
 * IncrementingWeightUpdater.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#include "IncrementingWeightUpdater.hpp"

namespace PV {

IncrementingWeightUpdater::IncrementingWeightUpdater(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void IncrementingWeightUpdater::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HebbianUpdater::initialize(name, params, comm);
}

int IncrementingWeightUpdater::updateWeights(int arborId) {
   long nPatch         = mWeights->getPatchSizeOverall();
   long numDataPatches = mWeights->getNumDataPatchesOverall();
   for (long patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
      float *Wdata  = mWeights->getDataFromDataIndex(arborId, patchIndex);
      float *dWdata = mDeltaWeights->getDataFromDataIndex(arborId, patchIndex);
      for (long k = 0; k < nPatch; k++) {
         float const dw = 1.0f;
         dWdata[k]      = dw;
         Wdata[k] += dw;
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
