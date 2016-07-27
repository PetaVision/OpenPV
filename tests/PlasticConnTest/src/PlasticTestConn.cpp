/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char * name, HyPerCol * hc) : HyPerConn(){
   HyPerConn::initialize(name, hc);
}

pvdata_t PlasticTestConn::updateRule_dW(pvdata_t pre, pvdata_t post) {
   return pre - post;
}

PlasticTestConn::~PlasticTestConn() {
}

} /* namespace PV */
