/*
 * MLPSigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "MLPSigmoidLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

// MLPSigmoidLayer can be used to implement Sigmoid junctions
namespace PV {
MLPSigmoidLayer::MLPSigmoidLayer() {
   initialize_base();
}

MLPSigmoidLayer::MLPSigmoidLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

MLPSigmoidLayer::~MLPSigmoidLayer()
{
   // Handled by CloneVLayer destructor
   // clayer->V = NULL;
   // free(sourceLayerName);
}

int MLPSigmoidLayer::initialize_base() {
   // Handled by CloneVLayer
   // sourceLayerName = NULL;
   // sourceLayer = NULL;
   linAlpha = 0;
   return PV_SUCCESS;
}

int MLPSigmoidLayer::initialize(const char * name, HyPerCol * hc) {
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

int MLPSigmoidLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_LinAlpha(ioFlag);
   return status;
}

void MLPSigmoidLayer::ioParam_LinAlpha(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "linAlpha", &linAlpha, linAlpha);
}

int MLPSigmoidLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   return status;
}

int MLPSigmoidLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
   assert(GSyn==NULL);
   return status;
}


int MLPSigmoidLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int MLPSigmoidLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), 0, NULL, linAlpha, getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

int MLPSigmoidLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, float linear_alpha, unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   setActivity_MLPSigmoidLayer(num_neurons, A, V, linear_alpha, nx, ny, nf, loc->nb, dt);
   // resetGSynBuffers(); // Since sourceLayer updates V, this->GSyn is not used
   return PV_SUCCESS;
}



} // end namespace PV

