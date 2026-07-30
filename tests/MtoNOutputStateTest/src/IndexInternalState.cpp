/**
 * IndexInternalState.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexInternalState.hpp"

namespace PV {

IndexInternalState::IndexInternalState(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

IndexInternalState::IndexInternalState() {}

IndexInternalState::~IndexInternalState() {}

void IndexInternalState::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void IndexInternalState::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parameters()->handleUnnecessaryStringParameter(getName(), "InitVType", nullptr);
}

PV::Response::Status
IndexInternalState::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = InternalStateBuffer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   updateBuffer(0.0 /*timestamp*/, message->mDeltaTime);
   return Response::SUCCESS;
}

void IndexInternalState::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   long const numNeurons  = (long)loc->nx * (long)loc->ny * (long)loc->nf;
   pvAssert(numNeurons == getBufferSize());
   long const numGlobalNeurons = (long)loc->nxGlobal * (long)loc->nyGlobal * (long)loc->nf;
   for (int b = 0; b < loc->nbatch; b++) {
      int const globalBatchIndex = b + loc->kb0;
      float *V                   = &mBufferData.data()[b * numNeurons];
      for (long k = 0; k < numNeurons; k++) {
         long kGlobal     = globalIndexFromLocal(k, *loc);
         long kGlobalBatch = kGlobal + globalBatchIndex * numGlobalNeurons;
         float value      = (float)kGlobalBatch * (float)simTime;
         V[k]             = value;
      }
   }
}

} // end namespace PV
