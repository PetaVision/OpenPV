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

DatastoreDelayTestProbe::DatastoreDelayTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize(name, hc);
}

int DatastoreDelayTestProbe::initialize(const char *name, HyPerCol *hc) {
   StatsProbe::initialize(name, hc);
   return PV_SUCCESS;
}

void DatastoreDelayTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      requireType(BufV);
   }
}

int DatastoreDelayTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = StatsProbe::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   HyPerLayer *inputLayer = message->lookup<HyPerLayer>(std::string("input"));
   FatalIf(inputLayer == nullptr, "Unable to find layer \"input\".\n");
   mNumDelayLevels = inputLayer->getNumDelayLevels();

   return status;
}

int DatastoreDelayTestProbe::outputState(double timed) {
   HyPerLayer *l = getTargetLayer();
   if (mOutputStreams.empty()) {
      return PV_SUCCESS;
   }
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
               timed,
               globalBatchIndex,
               (double)fMax[b],
               (int)correctValue);
         status = PV_FAILURE;
      }
      if (fMax[b] == correctValue and timed < mNumDelayLevels + 1) {
         output(0).printf(
               "%s: time %f: batch element %d has a neuron with value %f but should not reach it "
               "until time %d\n",
               l->getDescription_c(),
               timed,
               globalBatchIndex,
               (double)fMax[b],
               mNumDelayLevels + 1);
         status = PV_FAILURE;
      }
      if (fMin[b] < correctValue and timed >= mNumDelayLevels + 1) {
         output(b).printf(
               "%s: time %f, a neuron in batch element %d has value %f instead of %d.\n",
               l->getDescription_c(),
               timed,
               globalBatchIndex,
               (double)fMin[b],
               (int)correctValue);
         status = PV_FAILURE;
      }
   }

   FatalIf(status != PV_SUCCESS, "Test failed.\n");
   return PV_SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {}

} // end of namespace PV block
