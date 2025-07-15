#include "MaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include "MaxPoolTestBuffer.hpp"
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *MaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, MaxPoolTestBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
