/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"

namespace PV {

// This constructor allows derived classes to set an arbitrary number of channels
ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc, int numChannels) : ANNLayer(name, hc, numChannels) {
   //initialize();
   // TODO Auto-generated constructor stub

}

ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc, MAX_CHANNELS) {
   //initialize();
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNSquaredLayer::~ANNSquaredLayer()
{
   // TODO Auto-generated destructor stub
}

int ANNSquaredLayer::initialize() {
   return ANNLayer::initialize();
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
