#include "GateMaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include <components/ANNActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateMaxPoolTestLayer::GateMaxPoolTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *GateMaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GateMaxPoolTestBuffer,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
