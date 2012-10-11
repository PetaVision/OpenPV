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
   int status;
   status = updateState(timef, dt, &VInited, &nextUpdateTime, firstUpdateTime, displayPeriod, getLayerLoc(), getCLayer()->activity->data, getV(), getVprev(), getNumChannels(), GSyn[0], getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

int IncrementLayer::updateState(float timef, float dt, bool * inited, float * next_update_time, float first_update_time, float display_period, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * Vprev, int num_channels, pvdata_t * gSynHead, unsigned int * active_indices, unsigned int * num_active) {
   int status = PV_SUCCESS;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
//   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
//   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);

   if( *inited ) {
      if( timef >= *next_update_time ) {
         *next_update_time += display_period;
         pvdata_t * Vprev1 = Vprev;
         pvdata_t * V = getV();
         for( int k=0; k<num_neurons; k++ ) {
            *(Vprev1++) = *(V++);
         }
      }
      status = updateV_HyPerLayer(num_neurons, V, gSynHead);
      if( status == PV_SUCCESS ) status = setActivity_IncrementLayer(num_neurons, A, V, Vprev, nx, ny, nf, loc->nb); // setActivity();
      if( status == PV_SUCCESS ) status = resetGSynBuffers_HyPerLayer(num_neurons, num_channels, gSynHead); // resetGSynBuffers();
   }
   else {
      if( timef >= first_update_time ) {
         status = updateV_ANNLayer(num_neurons, V, gSynHead, max_pvdata_t, -max_pvdata_t, -max_pvdata_t); // updateV();
         resetGSynBuffers_HyPerLayer(num_neurons, num_channels, gSynHead); // resetGSynBuffers();
         *inited = true;
      }
   }
   return status;

}

int IncrementLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_IncrementLayer(getNumNeurons(), clayer->activity->data, getV(), getVprev(), loc->nx, loc->ny, loc->nf, loc->nb);
}

int IncrementLayer::checkpointRead(const char * cpDir, float * timef) {
   HyPerLayer::checkpointRead(cpDir, timef);
   InterColComm * icComm = parent->icCommunicator();
   double timed;
   char filename[PV_PATH_MAX];

   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_Vprev.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      if (icComm->commRank()==0) {
         fprintf(stderr, "IncrementLayer::checkpointRead error: path \"%s/%s_Vprev.pvp\" is too long.\n", cpDir, name);
      }
      abort();
   }
   readBufferFile(filename, icComm, &timed, Vprev, 1, /*extended*/false, /*contiguous*/false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   VInited = *timef >= firstUpdateTime + parent->getDeltaTime();

   return PV_SUCCESS;
}

int IncrementLayer::checkpointWrite(const char * cpDir) {
   HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+1+strlen(name)+12;
   // The +1 is for the slash between cpDir and name; the +12 needs to be large enough to hold the suffix (e.g. _G_Gap.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed = snprintf(filename, filenamesize, "%s/%s_Vprev.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, Vprev, 1, /*extended*/false, /*contiguous*/false, getLayerLoc()); // TODO contiguous=true
   free(filename);
   return PV_SUCCESS;
}


} /* namespace PV */
