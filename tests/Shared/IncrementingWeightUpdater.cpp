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
      Communicator *comm) {
   initialize(name, params, comm);
}

void IncrementingWeightUpdater::initialize(char const *name, PVParams *params, Communicator *comm) {
   HebbianUpdater::initialize(name, params, comm);
}

int IncrementingWeightUpdater::updateWeights(int arborId) {
   int nPatch         = mWeights->getPatchSizeOverall();
   int numDataPatches = mWeights->getNumDataPatches();
   for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
      float *Wdata  = mWeights->getDataFromDataIndex(arborId, patchIndex);
      float *dWdata = mDeltaWeights->getDataFromDataIndex(arborId, patchIndex);
      for (int k = 0; k < nPatch; k++) {
         Wdata[k] += 1.0f;
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
