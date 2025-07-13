/*
 * MomentumLCALayer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCALayer.hpp"
#include "components/MomentumLCAActivityComponent.hpp"

namespace PV {

MomentumLCALayer::MomentumLCALayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

MomentumLCALayer::~MomentumLCALayer() {}

void MomentumLCALayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLCALayer::initialize(params, defaults, comm);
}

ActivityComponent *MomentumLCALayer::createActivityComponent() {
   return new MomentumLCAActivityComponent(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
