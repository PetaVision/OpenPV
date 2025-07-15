/*
 * MomentumLCALayer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCALayer.hpp"
#include "components/MomentumLCAActivityComponent.hpp"

namespace PV {

MomentumLCALayer::MomentumLCALayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

MomentumLCALayer::~MomentumLCALayer() {}

void MomentumLCALayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLCALayer::initialize(paramsIO, comm);
}

ActivityComponent *MomentumLCALayer::createActivityComponent() {
   return new MomentumLCAActivityComponent(mParamsIO, mCommunicator);
}

} // end namespace PV
