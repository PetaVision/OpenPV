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

DatastoreDelayTestLayer::DatastoreDelayTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

DatastoreDelayTestLayer::~DatastoreDelayTestLayer() {}

void DatastoreDelayTestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

LayerInputBuffer *DatastoreDelayTestLayer::createLayerInput() { return nullptr; }

ActivityComponent *DatastoreDelayTestLayer::createActivityComponent() {
   return new CloneActivityComponent<DatastoreDelayTestBuffer, HyPerActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end of namespace PV block
