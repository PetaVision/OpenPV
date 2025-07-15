/*
 * DropoutLayer.cpp
 */

#include "DropoutLayer.hpp"
#include "components/DropoutActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

DropoutLayer::DropoutLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

DropoutLayer::~DropoutLayer() {}

void DropoutLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *DropoutLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     DropoutActivityBuffer>(mParamsIO, mCommunicator);
}

} // end namespace PV
