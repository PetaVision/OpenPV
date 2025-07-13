#include "PvpLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpActivityBuffer.hpp"

namespace PV {

PvpLayer::PvpLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PvpLayer::~PvpLayer() {}

void PvpLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InputLayer::initialize(params, defaults, comm);
}

ActivityComponent *PvpLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
