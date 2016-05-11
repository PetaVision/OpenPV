/*
 * MembranePotentialLayer.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: pschultz
 */

#include "MembranePotentialLayer.hpp"

namespace PV {

MembranePotentialLayer::MembranePotentialLayer() {
   initialize_base();
}

MembranePotentialLayer::MembranePotentialLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

MembranePotentialLayer::~MembranePotentialLayer() {
}

int MembranePotentialLayer::initialize_base() {
   return PV_SUCCESS;
}

int MembranePotentialLayer::initialize(const char * name, HyPerCol * hc) {
   return HyPerLayer::initialize(name, hc);
}

} /* namespace PV */
