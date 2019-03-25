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
   int const nPatch         = mWeights->getPatchSizeOverall();
   int const numDataPatches = mWeights->getNumDataPatches();
   for (int arbor = 0; arbor < mArborList->getNumAxonalArbors(); arbor++) {
      for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
         float *Wdata = mWeights->getDataFromDataIndex(arbor, patchIndex);
         for (int kPatch = 0; kPatch < nPatch; kPatch++) {
            Wdata[kPatch] = patchIndex * nPatch + kPatch + simTime;
         }
      }
   }
   mLastUpdateTime = simTime;
   mWeights->setTimestamp(simTime);
   computeNewWeightUpdateTime(simTime, mWeightUpdateTime);
}

} // namespace PV
