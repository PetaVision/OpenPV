#include "MovieTestLayer.hpp"

#include "MovieTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MovieTestLayer::MovieTestLayer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

MovieTestLayer::~MovieTestLayer() {}

ActivityComponent *MovieTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MovieTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
