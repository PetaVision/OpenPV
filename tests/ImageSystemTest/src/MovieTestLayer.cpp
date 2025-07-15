#include "MovieTestLayer.hpp"

#include "MovieTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MovieTestLayer::MovieTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

MovieTestLayer::~MovieTestLayer() {}

ActivityComponent *MovieTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MovieTestBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
