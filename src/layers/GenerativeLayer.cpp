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
   free( dAold );
   free( sparsitytermderivative );
}

int GenerativeLayer::initialize_base() {
   dAold = NULL;
   sparsitytermderivative = NULL;
   return PV_SUCCESS;
}

int GenerativeLayer::initialize(const char * name, HyPerCol * hc) {
   ANNLayer::initialize(name, hc, MAX_CHANNELS);
   PVParams * params = parent->parameters();
   relaxation = params->value(name, "relaxation", 1.0f);
   activityThreshold = params->value(name, "activityThreshold", 0.0f);
   auxChannelCoeff = params->value(name, "auxChannelCoeff", 0.0f);
   sparsityTermCoeff = params->value(name, "sparsityTermCoefficient", 1.0f);
   persistence = params->value(name, "persistence", 0.0f);
   dAold = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t *));
   if( dAold == NULL ) {
      fprintf(stderr, "Layer \"%s\": Unable to allocate memory for dAold\n", getName());
      exit(EXIT_FAILURE);
   }
   sparsitytermderivative = (pvdata_t *) malloc(getNumNeurons() * sizeof(pvdata_t *));
   if( sparsitytermderivative == NULL ) {
      fprintf(stderr, "Layer \"%s\": Unable to allocate memory for sparsitytermderivative\n", getName());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}  // end of GenerativeLayer::initialize()

int GenerativeLayer::updateState(float timef, float dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence, activityThreshold, getSpikingFlag(), getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) updateActiveIndices();
   return status;
}

int GenerativeLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence, pvdata_t activity_threshold, bool spiking, unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateSparsityTermDeriv_GenerativeLayer(num_neurons, V, sparsitytermderivative);
   updateV_GenerativeLayer(num_neurons, V, gSynHead, sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
   setActivity_GenerativeLayer(num_neurons, A, V, nx, ny, nf, loc->nb, activity_threshold);
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead);
   return PV_SUCCESS;
}

}  // end of namespace PV block

