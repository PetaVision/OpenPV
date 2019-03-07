/*
 * DatastoreDelayTestLayer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DatastoreDelayTestLayer.hpp"

#include "DatastoreDelayTestBuffer.hpp"
#include <components/CloneActivityComponent.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

DatastoreDelayTestLayer::DatastoreDelayTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

void DatastoreDelayTestLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *DatastoreDelayTestLayer::createLayerInput() { return nullptr; }

ActivityComponent *DatastoreDelayTestLayer::createActivityComponent() {
   return new CloneActivityComponent<DatastoreDelayTestBuffer, HyPerActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end of namespace PV block
