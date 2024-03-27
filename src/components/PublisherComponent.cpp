/*
 * PublisherComponent.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#include "PublisherComponent.hpp"

namespace PV {

PublisherComponent::PublisherComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PublisherComponent::PublisherComponent() {}

PublisherComponent::~PublisherComponent() {}

void PublisherComponent::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BasePublisherComponent::initialize(name, params, comm);
}

void PublisherComponent::setObjectType() { mObjectType = "PublisherComponent"; }

int PublisherComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_sparseLayer(ioFlag);
   return PV_SUCCESS;
}

void PublisherComponent::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "sparseLayer", &mSparseLayer, false);
}

} // namespace PV
