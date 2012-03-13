/*
 * LogLatWTAGenLayer.cpp
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAGenLayer.hpp"

namespace PV {

LogLatWTAGenLayer::LogLatWTAGenLayer() {
   initialize_base();
}

LogLatWTAGenLayer::LogLatWTAGenLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

LogLatWTAGenLayer::~LogLatWTAGenLayer() {}

int LogLatWTAGenLayer::initialize_base() {
   return PV_SUCCESS;
}

int LogLatWTAGenLayer::updateState(float timef, float dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence, activityThreshold, getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

int LogLatWTAGenLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence, pvdata_t activity_threshold, unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   pvdata_t * gSynAux = getChannelStart(gSynHead, CHANNEL_INHB, num_neurons);
   updateSparsityTermDeriv_LogLatWTAGenLayer(num_neurons, getLayerLoc()->nf, V, sparsitytermderivative);
   updateV_GenerativeLayer(num_neurons, V, gSynExc, gSynInh, gSynAux, sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
   setActivity_GenerativeLayer(num_neurons, A, V, nx, ny, nf, loc->nb, activity_threshold); // setActivity();
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();
   return PV_SUCCESS;
}

//int LogLatWTAGenLayer::updateSparsityTermDerivative() {
//   pvdata_t * V = getV();
//   int nf = getLayerLoc()->nf;
//   for( int k=0; k<getNumNeurons(); k+=nf) {
//      // Assumes that stride in features is one.
//      pvdata_t sumacrossfeatures = 0;
//      for( int f=0; f<nf; f++ ) {
//         sumacrossfeatures += V[k+f];
//      }
//      pvdata_t latWTAexpr = latWTAterm(V+k,nf); // a'*Lslash*a
//      for( int f=0; f<nf; f++ ) {
//         sparsitytermderivative[k+f] = 2*(sumacrossfeatures-V[k+f])/(1+latWTAexpr);
//      }
//   }
//   return PV_SUCCESS;
//}  // end of LogLatWTAGenLayer::updateSparsityTermDerivative()

//pvdata_t LogLatWTAGenLayer::latWTAterm(pvdata_t * V, int nf) {
//   pvdata_t z=0;
//   for( int p=0; p<nf; p++) {
//      for( int q=0; q<nf; q++) {
//         if( p!=q ) z += V[p]*V[q];
//      }
//   }
//    return z;
//}  // end of LogLatWTAGenLayer::latWTAterm(pvdata_t *, int)

}  // end of namespace PV block
