/*
 * DatastoreDelayTestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "DatastoreDelayTestProbe.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
DatastoreDelayTestProbe::DatastoreDelayTestProbe(const char * probeName, HyPerCol * hc) : StatsProbe()
{
   initDatastoreDelayTestProbe(probeName, hc);
}


int DatastoreDelayTestProbe::initDatastoreDelayTestProbe(const char * probeName, HyPerCol * hc) {
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
   HyPerLayer * l = getTargetLayer();
   InterColComm * icComm = l->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return PV_SUCCESS;
   }
   int status = PV_SUCCESS;
   int numDelayLevels = l->getParent()->getLayer(0)->getNumDelayLevels();
   pvdata_t correctValue = numDelayLevels*(numDelayLevels+1)/2;
   if( timed >= numDelayLevels+2 ) {
      pvdata_t * V = l->getV();
      for( int k=0; k<l->getNumNeuronsAllBatches(); k++ ) {
         if( V[k] != correctValue ) {
            fprintf(outputstream->fp, "Layer \"%s\": timef = %f, neuron %d: value is %f instead of %d\n", l->getName(), timed, k, V[k], (int) correctValue);
            status = PV_FAILURE;
         }
      }
      if( status == PV_SUCCESS) {
         fprintf(outputstream->fp, "Layer \"%s\": timef = %f, all neurons have correct value %d\n", l->getName(), timed, (int) correctValue);
      }
   }
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {
}

BaseObject * createDatastoreDelayTestProbe(char const * probeName, HyPerCol * hc) {
   return hc ? new DatastoreDelayTestProbe(probeName, hc) : NULL;
}

}  // end of namespace PV block
