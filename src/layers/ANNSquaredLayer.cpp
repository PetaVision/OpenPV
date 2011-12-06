/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"

namespace PV {

ANNSquaredLayer::ANNSquaredLayer() {
   initialize_base();
}

// This constructor allows derived classes to set an arbitrary number of channels
ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNSquaredLayer::~ANNSquaredLayer()
{
   // TODO Auto-generated destructor stub
}

int ANNSquaredLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNSquaredLayer::initialize(const char * name, HyPerCol * hc, int numChannels/*Default=MAX_CHANNELS*/) {
   return ANNLayer::initialize(name, hc, numChannels);
}

int ANNSquaredLayer::updateV() {
   ANNLayer::updateV();
   squareV();
   return PV_SUCCESS;
}

int ANNSquaredLayer::squareV() {
   pvdata_t * V = getV();
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] *= V[k];
   }
   return PV_SUCCESS;
}


} /* namespace PV */
