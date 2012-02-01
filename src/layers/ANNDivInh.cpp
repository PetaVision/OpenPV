/*
 * ANNDivInh.cpp
 *
 *  Created on: Jan 22, 2012
 *      Author: kpeterson
 */

#include "ANNDivInh.hpp"

namespace PV {

ANNDivInh::ANNDivInh()
{
   initialize_base();
}

ANNDivInh::ANNDivInh(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

ANNDivInh::ANNDivInh(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNDivInh::~ANNDivInh()
{
   // TODO Auto-generated destructor stub
}

int ANNDivInh::initialize_base() {
   return PV_SUCCESS;
}

int ANNDivInh::initialize(const char * name, HyPerCol * hc, int numChannels/*Default=MAX_CHANNELS*/) {
   return ANNLayer::initialize(name, hc, numChannels);
}

int ANNDivInh::updateV() {
//   ANNLayer::updateV();
//   squareV();
   pvdata_t * V = getV();
   pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
   pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
   pvdata_t * GSynDivInh = this->getChannel(CHANNEL_INHB);

   for( int k=0; k<getNumNeurons(); k++ ) {
      //V[k] = (GSynExc[k] - GSynInh[k])*(GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
//      printf("V[k] %f\n", V[k]);
//      printf("GSynExc[k] %f\n", GSynExc[k]);
//      printf("GSynInh[k] %f\n", GSynInh[k]);
//      printf("GSynDivInh[k] %f\n", GSynDivInh[k]);
      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
//      printf("after: V[k] %f\n", V[k]);
   }

   return PV_SUCCESS;
}

} /* namespace PV */
