/*
 * LCALayer.cpp
 *
 *  Created on: Sep 27, 2012
 *      Author: pschultz
 */

#include "LCALayer.hpp"

namespace PV {

LCALayer::LCALayer(const char * name, HyPerCol * hc, int num_channels) {
   initialize_base();
   initialize(name, hc, num_channels);
}

LCALayer::LCALayer() {
   initialize_base();
}

LCALayer::~LCALayer() {
}

int LCALayer::initialize_base() {
   return PV_SUCCESS;
}

int LCALayer::initialize(const char * name, HyPerCol * hc, int num_channels) {
   int status = HyPerLayer::initialize(name, hc, num_channels);
   threshold = readThreshold();
   thresholdSoftness = readThresholdSoftness();
   timeConstantTau = readTimeConstantTau();

   return status;
}

int LCALayer::updateState(float timef, float dt) {
   const pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
   const pvdata_t * gSynInh = getChannel(CHANNEL_INH);
   pvdata_t * V = getV();
   pvdata_t * A = getActivity();
   const float dt_tau = dt/timeConstantTau;
   const float threshdrop = thresholdSoftness * threshold;
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nb = loc->nb;
   for (int k=0; k<getNumNeurons(); k++) {
      pvdata_t stimulus = gSynExc[k] - gSynInh[k];
      pvdata_t Vk = V[k];
      Vk = Vk + dt_tau*(stimulus - V[k] + A[k]);
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      A[kex] = Vk >= threshold ? Vk - threshdrop : 0.0;
      V[k] = Vk;
   }

   return PV_SUCCESS;
}

} /* namespace PV */
