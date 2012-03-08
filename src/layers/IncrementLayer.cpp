/*
 * IncrementLayer.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: pschultz
 */

#include "IncrementLayer.hpp"

namespace PV {

IncrementLayer::IncrementLayer() {
   initialize_base();
}

IncrementLayer::IncrementLayer(const char* name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

IncrementLayer::~IncrementLayer() {
   free(Vprev);
}

int IncrementLayer::initialize_base() {
   Vprev = NULL;
   displayPeriod = 0;
   VInited = false;
   nextUpdateTime = 0;
   return PV_SUCCESS;
}

int IncrementLayer::initialize(const char* name, HyPerCol * hc, int numChannels) {
   int status = ANNLayer::initialize(name, hc, numChannels);
   displayPeriod = parent->parameters()->value(name, "displayPeriod", parent->getDeltaTime());
   firstUpdateTime = parent->parameters()->value(name, "firstUpdateTime", parent->simulationTime());
   nextUpdateTime = firstUpdateTime+displayPeriod;
   Vprev = (pvdata_t *) calloc(getNumNeurons(),sizeof(pvdata_t));
   if( Vprev == NULL ) {
      fprintf(stderr, "Unable to allocate Vprev buffer for IncrementLayer \"%s\"\n", name);
      abort();
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      assert(GSyn[0][k]==0 && GSyn[1][k]==0);
   }

   return status;
}

int IncrementLayer::readVThreshParams(PVParams * params) {
   // Threshold paramaters are not used, as updateState does not call applyVMax or applyVThresh
   // Override ANNLayer::readVThreshParams so that the threshold params are not read from the
   // params file, thereby creating an unnecessary warning.
   VMax = max_pvdata_t;
   VThresh = -max_pvdata_t;
   VMin = VThresh;
   return PV_SUCCESS;
}

int IncrementLayer::updateState(float timef, float dt) {
   return updateState(timef, dt, &VInited, &nextUpdateTime, firstUpdateTime, displayPeriod, getNumNeurons(), getV(), getVprev(), getChannel(CHANNEL_EXC), getChannel(CHANNEL_INH));
}

int IncrementLayer::updateState(float timef, float dt, bool * inited, float * next_update_time, float first_update_time, float display_period, int num_neurons, pvdata_t * V, pvdata_t * Vprev, pvdata_t * GSynExc, pvdata_t * GSynInh) {
   int status = PV_SUCCESS;
   if( *inited ) {
      if( timef >= *next_update_time ) {
         *next_update_time += display_period;
         pvdata_t * Vprev1 = Vprev;
         pvdata_t * V = getV();
         for( int k=0; k<num_neurons; k++ ) {
            *(Vprev1++) = *(V++);
         }
      }
      status = ANNLayer::updateState(timef, dt, num_neurons, V, GSynExc, GSynInh);
   }
   else {
      if( timef >= first_update_time ) {
         status = updateV_ANNLayer(num_neurons, V, GSynExc, GSynInh, max_pvdata_t, -max_pvdata_t, -max_pvdata_t); // updateV();
         resetGSynBuffers();
         *inited = true;
      }
   }
   return status;

}

int IncrementLayer::setActivity() {
   int status = PV_SUCCESS;
   for( int k=0; k<getNumNeurons(); k++ ) {
      clayer->activity->data[k] = getV()[k]-Vprev[k];
   }
   return status;
}

int IncrementLayer::checkpointRead(float * timef) {
   HyPerLayer::checkpointRead(timef);
   InterColComm * icComm = parent->icCommunicator();
   double timed;
   char * filename = (char *) malloc( (strlen(name)+12)*sizeof(char) );
   // The +12 needs to be large enough to hold the suffix (e.g. _G_IB.pvp) plus the null terminator
   assert(filename != NULL);

   sprintf(filename, "%s_Vprev.pvp", name);
   readBufferFile(filename, icComm, &timed, Vprev, 1, /*extended*/false, /*contiguous*/false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   VInited = *timef >= firstUpdateTime + parent->getDeltaTime();

   free(filename);
   return PV_SUCCESS;
}

int IncrementLayer::checkpointWrite() {
   HyPerLayer::checkpointWrite();
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   char * filename = (char *) malloc( (strlen(name)+12)*sizeof(char) );
   // The +12 needs to be large enough to hold the suffix (e.g. _G_IB.pvp) plus the null terminator
   assert(filename != NULL);
   sprintf(filename, "%s_Vprev.pvp", name);
   writeBufferFile(filename, icComm, timed, Vprev, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   free(filename);
   return PV_SUCCESS;
}


} /* namespace PV */
