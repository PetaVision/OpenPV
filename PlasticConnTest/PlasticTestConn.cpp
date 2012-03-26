/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel, const char * filename, InitWeights *weightInit) : KernelConn(){
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
}

int PlasticTestConn::update_dW(int axonId) {
   defaultUpdate_dW(axonId);

   return PV_SUCCESS;
}

pvdata_t PlasticTestConn::updateRule_dW(pvdata_t pre, pvdata_t post) {
   return pre - post;
}

PlasticTestConn::~PlasticTestConn() {
}

} /* namespace PV */
