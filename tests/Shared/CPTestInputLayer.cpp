/*
 * CPTestInputLayer.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "CPTestInputLayer.hpp"
#include "CPTest_updateStateFunctions.h"

void CPTestInputLayer_update_state(
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

CPTestInputLayer::CPTestInputLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

CPTestInputLayer::~CPTestInputLayer() {}

int CPTestInputLayer::initialize(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
   return PV_SUCCESS;
}

Response::Status CPTestInputLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   initializeV();
   return Response::SUCCESS;
}

void CPTestInputLayer::initializeV() {
   FatalIf(
         !(parent->parameters()->value(name, "restart", 0.0, false) == 0.0),
         "Test failed.\n"); // initializeV should only be called if restart is false
   const PVLayerLoc *loc = getLayerLoc();
   for (int b = 0; b < parent->getNBatch(); b++) {
      float *VBatch = getV() + b * getNumNeurons();
      for (int k = 0; k < getNumNeurons(); k++) {
         int kx = kxPos(k, loc->nx, loc->nx, loc->nf);
         int ky = kyPos(k, loc->nx, loc->ny, loc->nf);
         int kf = featureIndex(k, loc->nx, loc->ny, loc->nf);
         int kGlobal =
               kIndex(loc->kx0 + kx, loc->ky0 + ky, kf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         VBatch[k] = (float)kGlobal;
      }
   }
}

Response::Status CPTestInputLayer::updateState(double timed, double dt) {
   update_timer->start();
   const int nx         = clayer->loc.nx;
   const int ny         = clayer->loc.ny;
   const int nf         = clayer->loc.nf;
   const PVHalo *halo   = &clayer->loc.halo;
   const int numNeurons = getNumNeurons();
   const int nbatch     = clayer->loc.nbatch;

   float *GSynHead = GSyn[0];
   float *V        = getV();
   float *activity = clayer->activity->data;

   CPTestInputLayer_update_state(
         nbatch,
         numNeurons,
         nx,
         ny,
         nf,
         halo->lt,
         halo->rt,
         halo->dn,
         halo->up,
         V,
         GSynHead,
         activity);

   update_timer->stop();
   return Response::SUCCESS;
}

} // end of namespace PV block

// Kernel
void CPTestInputLayer_update_state(
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
   updateV_CPTestInputLayer(nbatch, numNeurons, V);
   setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   resetGSynBuffers_HyPerLayer(nbatch, numNeurons, 2, GSynHead);
}
