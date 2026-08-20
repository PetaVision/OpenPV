#include "GaussianNoiseLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/GaussianNoiseActivityBuffer.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"

namespace PV {

GaussianNoiseLayer::GaussianNoiseLayer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

GaussianNoiseLayer::~GaussianNoiseLayer() {}

void GaussianNoiseLayer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseLayer::initialize(name, params, comm);
}

ActivityComponent *GaussianNoiseLayer::createActivityComponent() {
   return new HyPerActivityComponent<HyPerInternalStateBuffer, GaussianNoiseActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
