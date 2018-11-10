#include "MoviePvpTestLayer.hpp"

#include "MoviePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MoviePvpTestLayer::MoviePvpTestLayer(char const *name, HyPerCol *hc) { initialize(name, hc); }

MoviePvpTestLayer::~MoviePvpTestLayer() {}

ActivityComponent *MoviePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MoviePvpTestBuffer>(getName(), parent);
}

} // end namespace PV
