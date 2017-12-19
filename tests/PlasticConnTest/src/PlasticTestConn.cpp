/*
 * PlasticTestConn.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestConn.hpp"
#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestConn::PlasticTestConn(const char *name, HyPerCol *hc) : HyPerConn() {
   HyPerConn::initialize(name, hc);
}

BaseWeightUpdater *PlasticTestConn::createWeightUpdater() {
   return new PlasticTestUpdater(name, parent);
}

PlasticTestConn::~PlasticTestConn() {}

} /* namespace PV */
