/*
 * PtwiseQuotientLayer.cpp
 *
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
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

int PtwiseQuotientLayer::initialize_base() {
   numChannels = 2;
   return PV_SUCCESS;
}

int PtwiseQuotientLayer::initialize(const char *name, HyPerCol *hc) {
   return ANNLayer::initialize(name, hc);
}

Response::Status PtwiseQuotientLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   pvAssert(numChannels >= 2);
   return status;
}

Response::Status PtwiseQuotientLayer::updateState(double timef, double dt) {
   doUpdateState(
         timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0]);
   return Response::SUCCESS;
}

void PtwiseQuotientLayer::doUpdateState(
      double timef,
      double dt,
      const PVLayerLoc *loc,
      float *A,
      float *V,
      int num_channels,
      float *gSynHead) {
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
