#include "GaussianNoiseLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/GaussianNoiseActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"

namespace PV {

GaussianNoiseLayer::GaussianNoiseLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

GaussianNoiseLayer::~GaussianNoiseLayer() {}

void GaussianNoiseLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *GaussianNoiseLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     GaussianNoiseActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
