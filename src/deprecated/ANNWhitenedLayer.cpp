/*
 * ANNWhitenedLayer.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: garkenyon
 */

// ANNWhitenedLayer was deprecated on Aug 15, 2018.

#include "ANNWhitenedLayer.hpp"
#include "DeprecatedUpdateStateFunctions.h"

void ANNWhitenedLayer_update_state(
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
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity);

namespace PV {

ANNWhitenedLayer::ANNWhitenedLayer() { initialize_base(); }

ANNWhitenedLayer::ANNWhitenedLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ANNWhitenedLayer::~ANNWhitenedLayer() {}

int ANNWhitenedLayer::initialize_base() { return PV_SUCCESS; }

void ANNWhitenedLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   WarnLog() << "ANNWhitenedLayer has been deprecated.\n";
   ANNLayer::initialize(name, params, comm);
   mLayerInput->requireChannel(2); // applyGSyn_ANNWhitenedLayer uses channels 0, 1, and 2
   pvAssert(mLayerInput->getNumChannels() == 3);
   return PV_SUCCESS;
}

Response::Status ANNWhitenedLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = mLayerInput->getLayerInput();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   ANNWhitenedLayer_update_state(
         nbatch,
         num_neurons,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         V,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         gSynHead,
         A);

   return Response::SUCCESS;
}

} /* namespace PV */

void ANNWhitenedLayer_update_state(
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
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity) {
   updateV_ANNWhitenedLayer(
         nbatch,
         numNeurons,
         V,
         GSynHead,
         activity,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up);
}
