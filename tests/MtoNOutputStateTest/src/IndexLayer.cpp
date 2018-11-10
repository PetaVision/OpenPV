/**
 * IndexLayer.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexLayer.hpp"

#include "IndexInternalState.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

IndexLayer::IndexLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

IndexLayer::~IndexLayer() {}

int IndexLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *IndexLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<IndexInternalState, HyPerActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
