#include "PvpLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpActivityBuffer.hpp"

namespace PV {

PvpLayer::PvpLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

PvpLayer::~PvpLayer() {}

void PvpLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InputLayer::initialize(paramsIO, comm);
}

ActivityComponent *PvpLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
