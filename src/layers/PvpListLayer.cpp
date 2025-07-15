#include "PvpListLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpListActivityBuffer.hpp"

namespace PV {

PvpListLayer::PvpListLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

PvpListLayer::~PvpListLayer() {}

void PvpListLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InputLayer::initialize(paramsIO, comm);
}

ActivityComponent *PvpListLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpListActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
