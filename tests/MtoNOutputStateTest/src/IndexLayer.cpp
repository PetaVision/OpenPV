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

PV::Response::Status
IndexLayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   double deltaTime = message->mDeltaTime;
   mLastUpdateTime  = deltaTime;
   mLastTriggerTime = deltaTime;
   return updateState(0.0 /*timestamp*/, deltaTime);
}

Response::Status IndexLayer::updateState(double timef, double dt) {
   PVLayerLoc const *loc = getLayerLoc();
   PVHalo const &halo    = loc->halo;
   for (int b = 0; b < loc->nbatch; b++) {
      float *V = &getV()[b * getNumNeurons()];
      float *A = &mActivity->getActivity()[b * getNumExtended()];
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
