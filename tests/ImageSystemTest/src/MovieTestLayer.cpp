#include "MovieTestLayer.hpp"

#include "MovieTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

MovieTestLayer::MovieTestLayer(char const *name, HyPerCol *hc) { initialize(name, hc); }

MovieTestLayer::~MovieTestLayer() {}

ActivityComponent *MovieTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MovieTestBuffer>(getName(), parent);
}

} // end namespace PV
