/*
 * AccumulateLayer.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: pschultz
 */

#include "AccumulateLayer.hpp"

namespace PV {

AccumulateLayer::AccumulateLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

AccumulateLayer::AccumulateLayer() {
   initialize_base();
}

int AccumulateLayer::initialize_base() {
   syncedInputLayerName = NULL;
   syncedInputLayer = NULL;
   return PV_SUCCESS;
}

int AccumulateLayer::initialize(const char * name, HyPerCol * hc) {
   return ANNLayer::initialize(name, hc);
}

int AccumulateLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_syncedInputLayer(ioFlag);
   return status;
}

void AccumulateLayer::ioParam_syncedInputLayer(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "syncedInputLayer", &syncedInputLayerName, NULL);
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
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
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
   //update_timer->start();
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      updateStateOpenCL(time, dt);
//      //HyPerLayer::updateState(time, dt);
//   }
//   else {
//#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      updateV_AccumulateLayer(num_neurons, V, num_channels, gSynHead, A,
              AMax, AMin, VThresh, AShift, VWidth, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      //Moved to publish
      //if (this->writeSparseActivity){
      //   updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      //}
//#ifdef PV_USE_OPENCL
//   }
//#endif

   //update_timer->stop();
   return PV_SUCCESS;
}

int AccumulateLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int status = PV_SUCCESS;
   memset(clayer->activity->data, 0, sizeof(pvdata_t)*getNumExtended());
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(num_neurons, getV(), AMin, VThresh, AShift, VWidth, getCLayer()->activity->data, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(num_neurons, getV(), AMax, getCLayer()->activity->data, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   return status;
}

AccumulateLayer::~AccumulateLayer() {
}

} /* namespace PV */
