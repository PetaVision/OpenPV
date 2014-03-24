/*
 * GenerativeLayer.cpp
 *
 * A class derived from ANNLayer where the update rule is
 * dAnew = (excitatorychannel - inhibitorychannel - log(1+old^2))
 * dAnew = persistenceOfMemory*dAold + (1-persistenceOfMemory)*dAnew
 * A = A + relaxation*dAnew
 * dAold = dAnew
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include "GenerativeLayer.hpp"

namespace PV {

GenerativeLayer::GenerativeLayer() {
   initialize_base();
}

GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

GenerativeLayer::~GenerativeLayer() {
   free( dV );
   free( sparsitytermderivative );
}

int GenerativeLayer::initialize_base() {
   numChannels = 3;
   dV = NULL;
   sparsitytermderivative = NULL;
   return PV_SUCCESS;
}

int GenerativeLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels == 3);
   return status;
}

int GenerativeLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_relaxation(ioFlag);
   ioParam_activityThreshold(ioFlag);
   ioParam_auxChannelCoeff(ioFlag);
   ioParam_sparsityTermCoefficient(ioFlag);
   ioParam_persistence(ioFlag);
   return PV_SUCCESS;
}  // end of GenerativeLayer::initialize()

void GenerativeLayer::ioParam_relaxation(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "relaxation", &relaxation, 1.0f/*default value*/);
}

void GenerativeLayer::ioParam_activityThreshold(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "activityThreshold", &activityThreshold, 0.0f/*default value*/);
}

void GenerativeLayer::ioParam_auxChannelCoeff(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "auxChannelCoeff", &auxChannelCoeff, 0.0f/*default value*/);
}

void GenerativeLayer::ioParam_sparsityTermCoefficient(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "sparsityTermCoefficient", &sparsityTermCoeff, 1.0f/*default value*/);
}

void GenerativeLayer::ioParam_persistence(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "persistence", &persistence, 0.0f/*default value*/);
}

int GenerativeLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   if (status != PV_SUCCESS) return status;
   if (status == PV_SUCCESS) status = allocateBuffer(&dV, getNumNeurons(), "dV");
   if (status == PV_SUCCESS) status = allocateBuffer(&sparsitytermderivative, getNumNeurons(), "sparsitytermderivative");
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
   return status;
}

int GenerativeLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], sparsitytermderivative, dV, AMax, AMin, VThresh, AShift, VWidth, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence, activityThreshold, getSpikingFlag(), getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) updateActiveIndices();
   return status;
}

int GenerativeLayer::updateState(double timef, double dt,
      const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels,
      pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dV,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth,
      pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff,
      pvdata_t persistence, pvdata_t activity_threshold, bool spiking,
      unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   pvdata_t relax_remaining = relaxation;
   while(relax_remaining > 0) {
      updateSparsityTermDeriv_GenerativeLayer(num_neurons, V, sparsitytermderivative);
      update_dV_GenerativeLayer(num_neurons, V, gSynHead,
            sparsitytermderivative, dV, AMax, AMin, VThresh,
            relax_remaining, auxChannelCoeff, sparsityTermCoeff,
            persistence);
      pvdata_t trunc_rel = reduce_relaxation(num_neurons, V, dV, relax_remaining);
      updateV_GenerativeLayer(num_neurons, V, dV, A, AMax, AMin, VThresh, AShift, VWidth, trunc_rel, nx, ny, nf, loc->nb);
      relax_remaining -=trunc_rel;
   }
   setActivity_GenerativeLayer(num_neurons, A, V, nx, ny, nf, loc->nb, activity_threshold);
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead);
   return PV_SUCCESS;
}

int GenerativeLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_GenerativeLayer(getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb, getActivityThreshold());
}

pvdata_t GenerativeLayer::reduce_relaxation(int num_neurons, pvdata_t * V, pvdata_t * dV, pvdata_t relaxation) {
   pvdata_t trunc_rel = relaxation;
   for( int k=0; k<num_neurons; k++) {
      if( dV[k] < 0 && V[k] > 0 ) {
         pvdata_t trunc_rel_test = -V[k]/dV[k];
         if( trunc_rel_test < trunc_rel ) trunc_rel = trunc_rel_test;
      }
   }
   return trunc_rel;
}

}  // end of namespace PV block

