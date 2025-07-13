#include "MovieTestLayer.hpp"

#include "MovieTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MovieTestLayer::MovieTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

MovieTestLayer::~MovieTestLayer() {}

ActivityComponent *MovieTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MovieTestBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
