/*
 * PublisherComponent.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#include "PublisherComponent.hpp"

namespace PV {

PublisherComponent::PublisherComponent(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

PublisherComponent::PublisherComponent() {}

PublisherComponent::~PublisherComponent() {}

void PublisherComponent::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BasePublisherComponent::initialize(paramsIO, comm);
}

void PublisherComponent::setObjectType() { mObjectType = "PublisherComponent"; }

int PublisherComponent::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_sparseLayer(ioSwitch);
   return PV_SUCCESS;
}

void PublisherComponent::ioParam_sparseLayer(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "sparseLayer", &mSparseLayerFlag);
}

} // namespace PV
