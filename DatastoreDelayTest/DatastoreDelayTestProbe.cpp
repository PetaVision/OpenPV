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
DatastoreDelayTestProbe::DatastoreDelayTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg) : StatsProbe()
{
   initDatastoreDelayTestProbe(probename, filename, layer, msg);
}


int DatastoreDelayTestProbe::initDatastoreDelayTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg) {
   initStatsProbe(filename, layer, BufActivity, msg);
   if( probename != NULL ) {
      name = strdup(probename);
   }
   else {
      name = NULL;
   }
   return PV_SUCCESS;
}
/**
 * @time
 * @l
 */
int DatastoreDelayTestProbe::outputState(float timef) {
   HyPerLayer * l = getTargetLayer();
#ifdef PV_USE_MPI
   InterColComm * icComm = l->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return PV_SUCCESS;
   }
#endif // PV_USE_MPI
   int status = PV_SUCCESS;
   int numDelayLevels = l->getParent()->getLayer(0)->getCLayer()->numDelayLevels;
   pvdata_t correctValue = numDelayLevels*(numDelayLevels+1)/2;
   if( timef >= numDelayLevels+2 ) {
      pvdata_t * V = l->getV();
      for( int k=0; k<l->getNumNeurons(); k++ ) {
         if( V[k] != correctValue ) {
            fprintf(fp, "Layer \"%s\": timef = %f, neuron %d: value is %f instead of %d\n", l->getName(), timef, k, V[k], (int) correctValue);
            status = PV_FAILURE;
         }
      }
      if( status == PV_SUCCESS) {
         fprintf(fp, "Layer \"%s\": timef = %f, all neurons have correct value %d\n", l->getName(), timef, (int) correctValue);
      }
   }
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
}

DatastoreDelayTestProbe::~DatastoreDelayTestProbe() {
   free(name);
}

}  // end of namespace PV block
