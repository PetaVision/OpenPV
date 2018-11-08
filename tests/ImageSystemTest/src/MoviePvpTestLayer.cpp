#include "MoviePvpTestLayer.hpp"

#include "MoviePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

MoviePvpTestLayer::~MoviePvpTestLayer() {}

ActivityComponent *MoviePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MoviePvpTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
