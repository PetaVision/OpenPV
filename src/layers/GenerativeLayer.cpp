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

/*
GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc, 3) {
   initialize_base();
   initialize();
}  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *)
 */
// No point to this constructor since PVLayerType doesn't get used
// GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type) : ANNLayer(name, hc){
//     initialize();
// }  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *, PVLayerType *)

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

//int GenerativeLayer::updateV() {
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
//   pvdata_t * GSynAux = this->getChannel(CHANNEL_INHB);
//   updateSparsityTermDerivative();
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      pvdata_t dAnew = GSynExc[k] - GSynInh[k] + auxChannelCoeff*GSynAux[k] - sparsityTermCoeff*sparsitytermderivative[k];
//      dAnew = persistence*dAold[k] + (1-persistence)*dAnew;
//      V[k] += relaxation*dAnew;
//      dAold[k] = dAnew;
//   }
//   applyVMax();
//   applyVThresh();
//   return PV_SUCCESS;
//}  // end of GenerativeLayer::updateV()

int GenerativeLayer::updateState(float timef, float dt) {
   return updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence, activityThreshold);
}

int GenerativeLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence, pvdata_t activity_threshold) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateSparsityTermDeriv_GenerativeLayer(num_neurons, V, sparsitytermderivative);
   pvdata_t * gSynExc = &gSynHead[((int) CHANNEL_EXC)*num_neurons];
   pvdata_t * gSynInh = &gSynHead[((int) CHANNEL_INH)*num_neurons];
   pvdata_t * gSynAux = &gSynHead[((int) CHANNEL_INHB)*num_neurons];
   updateV_GenerativeLayer(num_neurons, V, gSynExc, gSynInh, gSynAux, sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
   setActivity_GenerativeLayer(num_neurons, A, V, nx, ny, nf, loc->nb, activity_threshold); // setActivity();
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();
   updateActiveIndices();
   return PV_SUCCESS;
}

//int GenerativeLayer::updateSparsityTermDerivative() {
//   pvdata_t * V = getV();
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      pvdata_t vk = V[k];
//      sparsitytermderivative[k] = 2*vk/(1+vk*vk);
//   }
//   return PV_SUCCESS;
//}

//int GenerativeLayer::setActivity() {
//   const int nx = getLayerLoc()->nx;
//   const int ny = getLayerLoc()->ny;
//   const int nf = getLayerLoc()->nf;
//   const int marginWidth = getLayerLoc()->nb;
//   pvdata_t * activity = getCLayer()->activity->data;
//   pvdata_t * V = getV();
//   for( int k=0; k<getNumExtended(); k++ ) {
//      activity[k] = 0;
//   }
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      int kex = kIndexExtended( k, nx, ny, nf, marginWidth );
//      if( fabs(V[k]) > activityThreshold ) activity[kex] = V[k];
//      // fabs(V[k]) > activityThreshold ? activity[kex] : 0;
//   }
//   return PV_SUCCESS;
//}  // end of GenerativeLayer::setActivity()


}  // end of namespace PV block

