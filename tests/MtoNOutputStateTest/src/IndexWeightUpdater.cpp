/*
 * IndexWeightUpdater.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#include "IndexWeightUpdater.hpp"

namespace PV {

IndexWeightUpdater::IndexWeightUpdater(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void IndexWeightUpdater::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HebbianUpdater::initialize(name, params, comm);
}

Response::Status
IndexWeightUpdater::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   int const numArbors = mArborList->getNumAxonalArbors();
   updateState(0.0 /*simulationTime*/, message->mDeltaTime);
   for (int arbor = 0; arbor < numArbors; arbor++) {
      updateWeights(arbor);
   }
   return Response::SUCCESS;
}

void IndexWeightUpdater::updateState(double simTime, double dt) {
   long const nPatch         = mWeights->getPatchSizeOverall();
   long const numDataPatches = mWeights->getNumDataPatchesOverall();
   for (int arbor = 0; arbor < mArborList->getNumAxonalArbors(); arbor++) {
      for (long patchIndex = 0L; patchIndex < numDataPatches; patchIndex++) {
         float *Wdata = mWeights->getDataFromDataIndex(arbor, patchIndex);
         for (long kPatch = 0L; kPatch < nPatch; kPatch++) {
            Wdata[kPatch] = static_cast<long>(patchIndex * nPatch + kPatch) + simTime;
         }
      }
   }
   mLastUpdateTime = simTime;
   mWeights->setTimestamp(simTime);
   computeNewWeightUpdateTime(simTime, mWeightUpdateTime);
}

} // namespace PV
