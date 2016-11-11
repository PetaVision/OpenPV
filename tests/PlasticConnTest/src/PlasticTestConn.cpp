/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char *name, HyPerCol *hc) : HyPerConn() {
   HyPerConn::initialize(name, hc);
}

float PlasticTestConn::updateRule_dW(float pre, float post) { return pre - post; }

PlasticTestConn::~PlasticTestConn() {}

} /* namespace PV */
