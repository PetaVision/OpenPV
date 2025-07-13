#include "MoviePvpTestLayer.hpp"

#include "MoviePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

MoviePvpTestLayer::~MoviePvpTestLayer() {}

ActivityComponent *MoviePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MoviePvpTestBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
