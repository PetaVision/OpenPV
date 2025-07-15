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

ANNErrorLayer::ANNErrorLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ANNErrorLayer::~ANNErrorLayer() {}

void ANNErrorLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *ANNErrorLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     ErrScaleInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO, mCommunicator);
}

} // end namespace PV
