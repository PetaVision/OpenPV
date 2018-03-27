/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"

void ANNSquaredLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      float *GSynHead,
      float *activity);

namespace PV {

ANNSquaredLayer::ANNSquaredLayer() { initialize_base(); }

ANNSquaredLayer::ANNSquaredLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ANNSquaredLayer::~ANNSquaredLayer() {}

int ANNSquaredLayer::initialize_base() {
   numChannels = 1; // ANNSquaredLayer only takes input on the excitatory channel
   return PV_SUCCESS;
}

int ANNSquaredLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels == 1);
   return status;
}

Response::Status ANNSquaredLayer::updateState(double time, double dt) {
   const int nx     = clayer->loc.nx;
   const int ny     = clayer->loc.ny;
   const int nf     = clayer->loc.nf;
   const int nbatch = clayer->loc.nbatch;

   float *GSynHead = GSyn[0];
   float *V        = getV();
   float *activity = clayer->activity->data;

   ANNSquaredLayer_update_state(
         nbatch,
         getNumNeurons(),
         nx,
         ny,
         nf,
         clayer->loc.halo.lt,
         clayer->loc.halo.rt,
         clayer->loc.halo.dn,
         clayer->loc.halo.up,
         V,
         GSynHead,
         activity);
   return Response::SUCCESS;
}

} /* namespace PV */

///////////////////////////////////////////////////////
//
// implementation of ANNLayer kernels

void ANNSquaredLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      float *GSynHead,
      float *activity) {

   updateV_ANNSquaredLayer(nbatch, numNeurons, V, GSynHead);
   setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
}
