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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

void DatastoreDelayTestLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

LayerInputBuffer *DatastoreDelayTestLayer::createLayerInput() { return nullptr; }

ActivityComponent *DatastoreDelayTestLayer::createActivityComponent() {
   return new CloneActivityComponent<DatastoreDelayTestBuffer, HyPerActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end of namespace PV block
