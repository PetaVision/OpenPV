/*
 * DatastoreDelayTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "DatastoreDelayTestProbe.hpp"
#include "layers/HyPerLayer.hpp"
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

DatastoreDelayTestProbe::DatastoreDelayTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbe() {
   initialize(name, params, comm);
}

void DatastoreDelayTestProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

void DatastoreDelayTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      requireType(BufV);
   }
}

Response::Status DatastoreDelayTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = StatsProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   BasePublisherComponent *inputPublisher =
         message->mObjectTable->findObject<BasePublisherComponent>("input");
   pvAssert(inputPublisher);
   mNumDelayLevels = inputPublisher->getNumDelayLevels();

   return Response::SUCCESS;
}

Response::Status DatastoreDelayTestProbe::outputState(double simTime, double deltaTime) {
   if (mOutputStreams.empty()) {
      return Response::NO_ACTION;
   }
   HyPerLayer *l       = getTargetLayer();
   int status          = PV_SUCCESS;
   float correctValue  = mNumDelayLevels * (mNumDelayLevels + 1) / 2;
   int localBatchWidth = getTargetLayer()->getLayerLoc()->nbatch;
   int globalBatchOffset =
         localBatchWidth * (getMPIBlock()->getStartBatch() + getMPIBlock()->getBatchIndex());
   for (int b = 0; b < localBatchWidth; b++) {
      int globalBatchIndex = b + globalBatchOffset;
      if (fMax[b] > correctValue) {
         output(0).printf(
               "%s: time %f: batch element %d has a neuron with value %f but no neuron should ever "
               "get above %d\n",
               l->getDescription_c(),
               simTime,
               globalBatchIndex,
               (double)fMax[b],
               (int)correctValue);
         status = PV_FAILURE;
      }
      if (fMax[b] == correctValue and simTime < mNumDelayLevels + 1) {
         output(0).printf(
               "%s: time %f: batch element %d has a neuron with value %f but should not reach it "
               "until time %d\n",
               l->getDescription_c(),
               simTime,
               globalBatchIndex,
               (double)fMax[b],
               mNumDelayLevels + 1);
         status = PV_FAILURE;
      }
      if (fMin[b] < correctValue and simTime >= mNumDelayLevels + 1) {
         output(b).printf(
               "%s: time %f, a neuron in batch element %d has value %f instead of %d.\n",
               l->getDescription_c(),
               simTime,
               globalBatchIndex,
               (double)fMin[b],
               (int)correctValue);
         status = PV_FAILURE;
      }
   }

   FatalIf(status != PV_SUCCESS, "Test failed.\n");
   return Response::SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {}

} // end of namespace PV block
