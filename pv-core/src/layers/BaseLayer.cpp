/*
 * BaseLayer.cpp
 *
 *  Created on: Jan 16, 2010
 *      Author: Craig Rasmussen
 */

#include "BaseLayer.hpp"

namespace PV {

BaseLayer::BaseLayer() {
}

int BaseLayer::initialize(char const * name, HyPerCol * hc) {
   return BaseObject::initialize(name, hc);
}

BaseLayer::~BaseLayer() {
}

} // namespace PV


