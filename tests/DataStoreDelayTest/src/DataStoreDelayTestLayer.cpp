/*
 * DataStoreDelayTestLayer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DataStoreDelayTestLayer.hpp"

#include "DataStoreDelayTestBuffer.hpp"
#include <components/CloneActivityComponent.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

DataStoreDelayTestLayer::DataStoreDelayTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DataStoreDelayTestLayer::~DataStoreDelayTestLayer() {}

void DataStoreDelayTestLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *DataStoreDelayTestLayer::createLayerInput() { return nullptr; }

ActivityComponent *DataStoreDelayTestLayer::createActivityComponent() {
   return new CloneActivityComponent<DataStoreDelayTestBuffer, HyPerActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end of namespace PV block
