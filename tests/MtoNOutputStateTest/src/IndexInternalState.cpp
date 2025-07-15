/**
 * IndexInternalState.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexInternalState.hpp"

namespace PV {

IndexInternalState::IndexInternalState(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

IndexInternalState::IndexInternalState() {}

IndexInternalState::~IndexInternalState() {}

void IndexInternalState::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   InternalStateBuffer::initialize(paramsIO, comm);
}

void IndexInternalState::ioParam_InitVType(ParamsIOSwitch ioSwitch) {
   mParamsIO->handleUnnecessaryParameter("InitVType", std::string(""));
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
   int const numNeurons  = loc->nx * loc->ny * loc->nf;
   pvAssert(numNeurons == getBufferSize());
   int const numGlobalNeurons = loc->nxGlobal * loc->nyGlobal * loc->nf;
   for (int b = 0; b < loc->nbatch; b++) {
      int const globalBatchIndex = b + loc->kb0;
      float *V                   = &mBufferData.data()[b * numNeurons];
      for (int k = 0; k < numNeurons; k++) {
         int kGlobal      = globalIndexFromLocal(k, *loc);
         int kGlobalBatch = kGlobal + globalBatchIndex * numGlobalNeurons;
         float value      = (float)kGlobalBatch * (float)simTime;
         V[k]             = value;
      }
   }
}

} // end namespace PV
