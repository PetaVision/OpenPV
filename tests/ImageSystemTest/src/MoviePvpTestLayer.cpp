#include "MoviePvpTestLayer.hpp"

#include "MoviePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

MoviePvpTestLayer::~MoviePvpTestLayer() {}

ActivityComponent *MoviePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MoviePvpTestBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
