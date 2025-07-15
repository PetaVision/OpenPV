#include "GaussianNoiseLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/GaussianNoiseActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"

namespace PV {

GaussianNoiseLayer::GaussianNoiseLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

GaussianNoiseLayer::~GaussianNoiseLayer() {}

void GaussianNoiseLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *GaussianNoiseLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     GaussianNoiseActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
