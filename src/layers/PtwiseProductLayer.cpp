/*
 * PtwiseProductLayer.cpp
 *
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseProductLayer.hpp"

namespace PV {

PtwiseProductLayer::PtwiseProductLayer() {
   initialize_base();
}

PtwiseProductLayer::PtwiseProductLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {
}

int PtwiseProductLayer::initialize_base() {
   numChannels = 2;
   return PV_SUCCESS;
}

int PtwiseProductLayer::initialize(const char * name, HyPerCol * hc) {
   return ANNLayer::initialize(name, hc);
}

int PtwiseProductLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   assert(numChannels>=2);
   return status;
}

int PtwiseProductLayer::updateState(double timef, double dt) {
   int status;
   status = doUpdateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0]);
   return status;
}

int PtwiseProductLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
//   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
//   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   updateV_PtwiseProductLayer(num_neurons, V, gSynHead);
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // setActivity();
   //resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();
   return PV_SUCCESS;
}

//int PtwiseProductLayer::updateV() {
//    pvdata_t * V = getV();
//    pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
//    pvdata_t * GSynInh = getChannel(CHANNEL_INH);
//    for( int k=0; k<getNumNeurons(); k++ ) {
//        V[k] = GSynExc[k] * GSynInh[k];
//    }
//    return PV_SUCCESS;
//}  // end PtwiseProductLayer::updateV()

}  // end namespace PV
