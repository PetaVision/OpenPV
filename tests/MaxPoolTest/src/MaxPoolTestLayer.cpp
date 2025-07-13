#include "MaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include "MaxPoolTestBuffer.hpp"
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *MaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, MaxPoolTestBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
