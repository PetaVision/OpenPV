/*
 * DenseLayerOutputComponent.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#include "DenseLayerOutputComponent.hpp"

namespace PV {

DenseLayerOutputComponent::DenseLayerOutputComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

DenseLayerOutputComponent::DenseLayerOutputComponent() {}

DenseLayerOutputComponent::~DenseLayerOutputComponent() {}

void DenseLayerOutputComponent::initialize(char const *name, PVParams *params, Communicator *comm) {
   LayerOutputComponent::initialize(name, params, comm);
   mSparseLayer = false;
}

void DenseLayerOutputComponent::setObjectType() { mObjectType = "DenseLayerOutputComponent"; }

void DenseLayerOutputComponent::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sparseLayer", &mSparseLayer, false);
}

} // namespace PV
