/*
 * BIDSLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 */

#include "HyPerLayer.hpp"
#include "BIDSLayer.hpp"
#include "LIF.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {
BIDSLayer::BIDSLayer() {
  initialize_base();
  // initialize(arguments) should *not* be called by the protected constructor.
}

BIDSLayer::BIDSLayer(const char * name, HyPerCol * hc) {
  initialize_base();
  initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
}

int BIDSLayer::initialize_base() {
  // the most basic initializations.  Don't call any virtual methods,
  // or methods that call virtual methods, etc. from initialize_base();
   return PV_SUCCESS;
}

int BIDSLayer::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name) {
  // DerivedLayer-specific initializations that need to precede BaseClass initialization, if any
  LIF::initialize(name, hc, type, num_channels, kernel_name);
  // DerivedLayer-specific initializations
  return PV_SUCCESS;
}

int BIDSLayer::updateState(float timef, float dt){
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], getSpikingFlag(), getCLayer()->activeIndices, &getCLayer()->numActive);
   if(status == PV_SUCCESS) status = updateActiveIndices();
   return status;
}

int BIDSLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active)
{
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   //pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   updateV_HyPerLayer(num_neurons, V, gSynHead);
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->nb);
   // setActivity();
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();

   return PV_SUCCESS;
}
  // other DerivedLayer methods
}
