#include "PvpListLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpListActivityBuffer.hpp"

namespace PV {

PvpListLayer::PvpListLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PvpListLayer::~PvpListLayer() {}

void PvpListLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InputLayer::initialize(params, defaults, comm);
}

ActivityComponent *PvpListLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpListActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
