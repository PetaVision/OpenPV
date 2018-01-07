/*
 * IndexWeightUpdater.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#include "IndexWeightUpdater.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

IndexWeightUpdater::IndexWeightUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int IndexWeightUpdater::initialize(char const *name, HyPerCol *hc) {
   return HebbianUpdater::initialize(name, hc);
}

void IndexWeightUpdater::initializeWeights() {
   int const numArbors = mArborList->getNumAxonalArbors();
   int status          = PV_SUCCESS;
   for (int arbor = 0; arbor < numArbors; arbor++) {
      updateWeights(arbor);
   }
}

int IndexWeightUpdater::updateWeights(int arborId) {
   int const nPatch         = mWeights->getPatchSizeOverall();
   int const numDataPatches = mWeights->getNumDataPatches();
   for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
      float *Wdata = mWeights->getDataFromDataIndex(arborId, patchIndex);
      for (int kPatch = 0; kPatch < nPatch; kPatch++) {
         Wdata[kPatch] = patchIndex * nPatch + kPatch + parent->simulationTime();
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
