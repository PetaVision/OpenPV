/*
 * SparseLayerFlagPublisherComponent.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#include "SparseLayerFlagPublisherComponent.hpp"

namespace PV {

SparseLayerFlagPublisherComponent::SparseLayerFlagPublisherComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

SparseLayerFlagPublisherComponent::SparseLayerFlagPublisherComponent() {}

SparseLayerFlagPublisherComponent::~SparseLayerFlagPublisherComponent() {}

void SparseLayerFlagPublisherComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   PublisherComponent::initialize(name, params, comm);
}

void SparseLayerFlagPublisherComponent::setObjectType() {
   mObjectType = "SparseLayerFlagPublisherComponent";
}

int SparseLayerFlagPublisherComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_sparseLayer(ioFlag);
   return PV_SUCCESS;
}

void SparseLayerFlagPublisherComponent::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sparseLayer", &mSparseLayer, false);
}

} // namespace PV
