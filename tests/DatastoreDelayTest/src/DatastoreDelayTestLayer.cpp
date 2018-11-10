/*
 * DatastoreDelayTestLayer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DatastoreDelayTestLayer.hpp"

#include "DatastoreDelayTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

DatastoreDelayTestLayer::DatastoreDelayTestLayer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

int DatastoreDelayTestLayer::initialize(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
   return PV_SUCCESS;
}

LayerInputBuffer *DatastoreDelayTestLayer::createLayerInput() { return nullptr; }

ActivityComponent *DatastoreDelayTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<DatastoreDelayTestBuffer, HyPerActivityBuffer>(
         getName(), parent);
}

} // end of namespace PV block
