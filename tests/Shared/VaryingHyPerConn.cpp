/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"
#include "IncrementingWeightUpdater.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(const char *name, HyPerCol *hc) : HyPerConn() {
   initialize(name, hc);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

int VaryingHyPerConn::initialize(const char *name, HyPerCol *hc) {
   return HyPerConn::initialize(name, hc);
}

BaseWeightUpdater *VaryingHyPerConn::createWeightUpdater() {
   return new IncrementingWeightUpdater(name, parent);
}

} // end of namespace PV block
