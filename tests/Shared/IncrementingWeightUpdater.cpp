/*
 * IncrementingWeightUpdater.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#include "IncrementingWeightUpdater.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

IncrementingWeightUpdater::IncrementingWeightUpdater(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

int IncrementingWeightUpdater::initialize(char const *name, HyPerCol *hc) {
   return HebbianUpdater::initialize(name, hc);
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
