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

PtwiseProductLayer::PtwiseProductLayer() { initialize_base(); }

PtwiseProductLayer::PtwiseProductLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
} // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {}

int PtwiseProductLayer::initialize_base() { return PV_SUCCESS; }

int PtwiseProductLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   mLayerInput->requireChannel(0);
   mLayerInput->requireChannel(1);
   return status;
}

Response::Status PtwiseProductLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   pvAssert(mLayerInput->getNumChannels() >= 2);
   return status;
}

Response::Status PtwiseProductLayer::updateState(double timef, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
   int num_channels      = getNumChannels();
   float const *gSynHead = mLayerInput->getBufferData();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   updateV_PtwiseProductLayer(nbatch, num_neurons, V, gSynHead);
   setActivity_HyPerLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);
   return Response::SUCCESS;
}

} // end namespace PV
