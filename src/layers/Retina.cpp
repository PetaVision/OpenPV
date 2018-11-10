/*
 * Retina.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "Retina.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RetinaActivityBuffer.hpp"

namespace PV {

Retina::Retina(const char *name, HyPerCol *hc) { initialize(name, hc); }

Retina::Retina() {}

Retina::~Retina() {}

int Retina::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *Retina::createActivityComponent() {
   return new ActivityComponentActivityOnly<RetinaActivityBuffer>(getName(), parent);
}

} // namespace PV
