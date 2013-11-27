/*
 * AccumulateLayer.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: pschultz
 */

#include "AccumulateLayer.hpp"

namespace PV {

AccumulateLayer::AccumulateLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

AccumulateLayer::AccumulateLayer() {
   initialize_base();
}

int AccumulateLayer::initialize_base() {
   syncedInputLayerName = NULL;
   syncedInputLayer = NULL;
   return PV_SUCCESS;
}

int AccumulateLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   return ANNLayer::initialize(name, hc, numChannels);
}

int AccumulateLayer::setParams(PVParams * inputParams) {
   int status = ANNLayer::setParams(inputParams);
   readSyncedInputLayer(inputParams);
   return status;
}

void AccumulateLayer::readSyncedInputLayer(PVParams * params) {
   const char * synced_input_layer_name = params->stringValue(name, "syncedInputLayer");
   if (synced_input_layer_name != NULL) {
      syncedInputLayerName = strdup(synced_input_layer_name);
      if (syncedInputLayerName == NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate memory for syncedInputLayer: %s\n", params->groupKeywordFromName(name), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
}

int AccumulateLayer::communicateInitInfo() {
   if (syncedInputLayerName != NULL) {
      syncedInputLayer = parent->getLayerFromName(syncedInputLayerName);
      if (syncedInputLayer == NULL) {
         fprintf(stderr, "%s \"%s\": syncedInputLayer \"%s\" is not a layer in the column.\n", parent->parameters()->groupKeywordFromName(name), name, syncedInputLayerName);
         exit(EXIT_FAILURE);
      }
   }
   return PV_SUCCESS;
}

int AccumulateLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   bool needsUpdate = false;
   if (syncedInputLayer != NULL) {
      if (getPhase() > syncedInputLayer->getPhase()) {
         needsUpdate = syncedInputLayer->getLastUpdateTime() >= lastUpdateTime;
      }
      else {
         needsUpdate = syncedInputLayer->getLastUpdateTime() > lastUpdateTime;
      }
   }
   if (needsUpdate) {
      memset(clayer->activity->data, 0, sizeof(pvdata_t)*getNumExtended());
   }
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      updateV_AccumulateLayer(num_neurons, V, num_channels, gSynHead, A,
              VMax, VMin, VThresh, VShift, VWidth, nx, ny, nf, loc->nb);
      if (this->writeSparseActivity){
         updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      }
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

int AccumulateLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nb = loc->nb;
   int num_neurons = nx*ny*nf;
   int status;
   memset(clayer->activity->data, 0, sizeof(pvdata_t)*getNumExtended());
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(num_neurons, getV(), VMin, VThresh, VShift, VWidth, getCLayer()->activity->data, nx, ny, nf, nb);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(num_neurons, getV(), VMax, getCLayer()->activity->data, nx, ny, nf, nb);
   return status;
}

AccumulateLayer::~AccumulateLayer() {
}

} /* namespace PV */
