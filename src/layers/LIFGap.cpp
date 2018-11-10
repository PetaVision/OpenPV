/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "components/LIFGapActivityComponent.hpp"

namespace PV {

LIFGap::LIFGap(const char *name, HyPerCol *hc) { initialize(name, hc); }

LIFGap::LIFGap() {}

LIFGap::~LIFGap() {}

int LIFGap::initialize(const char *name, HyPerCol *hc) { return LIF::initialize(name, hc); }

ActivityComponent *LIFGap::createActivityComponent() {
   return new LIFGapActivityComponent(getName(), parent);
}

} // end namespace PV
