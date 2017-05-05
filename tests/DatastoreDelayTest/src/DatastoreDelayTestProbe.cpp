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

/**
 * @time
 * @l
 */
int DatastoreDelayTestProbe::outputState(double timed) {
   HyPerLayer *l        = getTargetLayer();
   Communicator *icComm = l->getParent()->getCommunicator();
   const int rcvProc    = 0;
   if (icComm->commRank() != rcvProc) {
      return PV_SUCCESS;
   }
   int status         = PV_SUCCESS;
   int numDelayLevels = l->getParent()->getLayer(0)->getNumDelayLevels();
   float correctValue = numDelayLevels * (numDelayLevels + 1) / 2;
   for (int k = 0; k < l->getNumNeuronsAllBatches(); k++) {
      float *V = l->getV();
      for (int k = 0; k < l->getNumNeuronsAllBatches(); k++) {
         float v = V[k];
         if (v < correctValue) {
            if (timed >= numDelayLevels + 1) {
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
            if (timed < numDelayLevels + 1) {
               outputStream->printf(
                     "%s: time %f, neuron %d has value %f, but should not reach it until %d\n",
                     l->getDescription_c(),
                     timed,
                     k,
                     (double)v,
                     numDelayLevels + 1);
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
