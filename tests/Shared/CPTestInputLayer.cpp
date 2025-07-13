/*
 * CPTestInputLayer.cpp
 */

#include "CPTestInputLayer.hpp"
#include "CPTestInputInternalStateBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>

namespace PV {

CPTestInputLayer::CPTestInputLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

CPTestInputLayer::~CPTestInputLayer() {}

void CPTestInputLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *CPTestInputLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     CPTestInputInternalStateBuffer,
                                     HyPerActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
