/*
 * PtwiseQuotientLayer.cpp
 *
 * The output V is the pointwise division of GSynExc by GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseQuotientLayer.hpp"

namespace PV {

PtwiseQuotientLayer::PtwiseQuotientLayer() { initialize_base(); }

PtwiseQuotientLayer::PtwiseQuotientLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
} // end PtwiseQuotientLayer::PtwiseQuotientLayer(const char *, HyPerCol *)

PtwiseQuotientLayer::~PtwiseQuotientLayer() {}

int PtwiseQuotientLayer::initialize_base() { return PV_SUCCESS; }

int PtwiseQuotientLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   mLayerInput->requireChannel(0);
   mLayerInput->requireChannel(1);
   return status;
}

Response::Status PtwiseQuotientLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   pvAssert(mLayerInput->getNumChannels() >= 2);
   return status;
}

Response::Status PtwiseQuotientLayer::updateState(double timef, double dt) {
   doUpdateState(
         timef,
         dt,
         getLayerLoc(),
         getActivity(),
         getV(),
         getNumChannels(),
         mLayerInput->getBufferData());
   return Response::SUCCESS;
}

void PtwiseQuotientLayer::doUpdateState(
      double timef,
      double dt,
      const PVLayerLoc *loc,
      float *A,
      float *V,
      int num_channels,
      float const *gSynHead) {
   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int num_neurons = nx * ny * nf;
   int nbatch      = loc->nbatch;
   updateV_PtwiseQuotientLayer(nbatch, num_neurons, V, gSynHead);
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
}

} // end namespace PV
