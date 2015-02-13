/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) : HyPerConn(){
   HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
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
