/*
 * LeakyIntegratedCountLayer.cpp
 *
 *  Created on: Dec 6, 2012
 *      Author: pschultz
 */

#include "LeakyIntegratedCountLayer.hpp"

namespace PV {

LeakyIntegratedCountLayer::LeakyIntegratedCountLayer() {
   initialize_base();
}

LeakyIntegratedCountLayer::LeakyIntegratedCountLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int LeakyIntegratedCountLayer::initialize_base() {
   decayTime = 0.0;
   return PV_SUCCESS;
}

int LeakyIntegratedCountLayer::initialize(const char * name, HyPerCol * hc) {
   ANNLayer::initialize(name, hc, 1); // Only need excitatory channel

   decayTime = readIntegrationTime();


   return PV_SUCCESS;
}

int LeakyIntegratedCountLayer::readVThreshParams(PVParams * params) {
   VMax = max_pvdata_t;
   VMin = -max_pvdata_t;
   VThresh = -max_pvdata_t;

   return PV_SUCCESS;
}

int LeakyIntegratedCountLayer::updateState(double timed, double dt) {
   pvdata_t * V = getV();
   pvdata_t * GSynExc = GSyn[0];
   pvdata_t * A = getActivity();
   double decayfactor = exp(-parent->getDeltaTime()/getDecayTime());
   for (int k=0; k<getNumNeurons(); k++) {
      V[k] = (pvdata_t) (decayfactor*(double) (V[k]+GSynExc[k]));
      A[k] = V[k];
      GSynExc[k] = 0.0f;
   }
   return PV_SUCCESS;
}

LeakyIntegratedCountLayer::~LeakyIntegratedCountLayer() {
}

} // end namespace PV
