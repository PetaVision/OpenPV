/*
 * DatastoreDelayTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "DatastoreDelayTestProbe.hpp"
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
DatastoreDelayTestProbe::DatastoreDelayTestProbe(const char *probeName, HyPerCol *hc)
      : StatsProbe() {
   initDatastoreDelayTestProbe(probeName, hc);
}

int DatastoreDelayTestProbe::initDatastoreDelayTestProbe(const char *probeName, HyPerCol *hc) {
   initStatsProbe(probeName, hc);
   return PV_SUCCESS;
}

void DatastoreDelayTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      requireType(BufActivity);
   }
}

int DatastoreDelayTestProbe::communicateInitInfo(CommunicateInitInfoMessage const *message) {
   int status = StatsProbe::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   Observer *inputObject  = message->lookup(std::string("input"));
   HyPerLayer *inputLayer = dynamic_cast<HyPerLayer *>(inputObject);
   FatalIf(inputLayer == nullptr, "Unable to find layer \"input\".\n");
   mNumDelayLevels = inputLayer->getNumDelayLevels();

   return status;
}

/**
 * @time
 * @l
 */
int DatastoreDelayTestProbe::outputState(double timed) {
   HyPerLayer *l        = getTargetLayer();
   Communicator *icComm = parent->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return PV_SUCCESS;
   }
   int status         = PV_SUCCESS;
   float correctValue = mNumDelayLevels * (mNumDelayLevels + 1) / 2;
   for (int k = 0; k < l->getNumNeuronsAllBatches(); k++) {
      float *V = l->getV();
      for (int k = 0; k < l->getNumNeuronsAllBatches(); k++) {
         float v = V[k];
         if (v < correctValue) {
            if (timed >= mNumDelayLevels + 1) {
               outputStream->printf(
                     "%s: time %f, neuron %d: value is %f instead of %d\n",
                     l->getDescription_c(),
                     timed,
                     k,
                     (double)V[k],
                     (int)correctValue);
               status = PV_FAILURE;
            }
         }
         else if (v == correctValue) {
            if (timed < mNumDelayLevels + 1) {
               outputStream->printf(
                     "%s: time %f, neuron %d has value %f, but should not reach it until %d\n",
                     l->getDescription_c(),
                     timed,
                     k,
                     (double)v,
                     mNumDelayLevels + 1);
               status = PV_FAILURE;
            }
         }
         else { // v > correctValue
            outputStream->printf(
                  "%s: time %f, neuron %d: value is %f but no neuron should ever get above %d\n",
                  l->getDescription_c(),
                  timed,
                  k,
                  (double)v,
                  (int)correctValue);
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
   return PV_SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {}

} // end of namespace PV block
