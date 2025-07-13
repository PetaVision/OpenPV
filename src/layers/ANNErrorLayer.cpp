/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ErrScaleInternalStateBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"

namespace PV {

ANNErrorLayer::ANNErrorLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ANNErrorLayer::~ANNErrorLayer() {}

void ANNErrorLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *ANNErrorLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     ErrScaleInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
