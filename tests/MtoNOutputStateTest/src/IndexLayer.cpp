/**
 * IndexLayer.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexLayer.hpp"

namespace PV {

IndexLayer::IndexLayer(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

IndexLayer::IndexLayer() { initialize_base(); }

IndexLayer::~IndexLayer() {}

int IndexLayer::initialize_base() { return PV_SUCCESS; }

int IndexLayer::initialize(char const *name, HyPerCol *hc) {
   return HyPerLayer::initialize(name, hc);
}

void IndexLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   mInitVObject = nullptr;
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

PV::Response::Status IndexLayer::initializeState() {
   return updateState(0.0 /*timestamp*/, parent->getDeltaTime());
}

Response::Status IndexLayer::updateState(double timef, double dt) {
   PVLayerLoc const *loc = getLayerLoc();
   PVHalo const &halo    = loc->halo;
   for (int b = 0; b < loc->nbatch; b++) {
      float *V = &clayer->V[b * getNumNeurons()];
      float *A = &clayer->activity->data[b * getNumExtended()];
      for (int k = 0; k < getNumNeurons(); k++) {
         int kGlobal      = globalIndexFromLocal(k, *loc);
         int kGlobalBatch = kGlobal + (b + loc->kb0) * getNumGlobalNeurons();
         float value      = (float)kGlobalBatch * (float)timef;
         V[k]             = value;
         int kExt =
               kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo.lt, halo.rt, halo.dn, halo.up);
         A[kExt] = value;
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
