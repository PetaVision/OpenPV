/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

ConstantLayer::ConstantLayer() {}

ConstantLayer::~ConstantLayer() {}

void ConstantLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

LayerUpdateController *ConstantLayer::createLayerUpdateController() { return nullptr; }

} /* namespace PV */
