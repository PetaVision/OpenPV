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
   return status;
}

int LCALayer::updateState(float timef, float dt) {
   pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
   pvdata_t * gSynInh = getChannel(CHANNEL_INH);
   for (int k=0; k<getNumNeurons(); k++) {
      pvdata_t stimulus = gSynExc[k] - gSynInh[k];

   }

   return PV_SUCCESS;
}

} /* namespace PV */
