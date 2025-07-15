/*
 * CPTestInputLayer.cpp
 */

#include "CPTestInputLayer.hpp"
#include "CPTestInputInternalStateBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>

namespace PV {

CPTestInputLayer::CPTestInputLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

CPTestInputLayer::~CPTestInputLayer() {}

void CPTestInputLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *CPTestInputLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     CPTestInputInternalStateBuffer,
                                     HyPerActivityBuffer>(mParamsIO, mCommunicator);
}

} // end namespace PV
