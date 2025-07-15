#include "GateMaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include <components/ANNActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateMaxPoolTestLayer::GateMaxPoolTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *GateMaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GateMaxPoolTestBuffer,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO, mCommunicator);
}

} /* namespace PV */
