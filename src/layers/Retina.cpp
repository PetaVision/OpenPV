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

Retina::Retina(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

Retina::Retina() {}

Retina::~Retina() {}

void Retina::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *Retina::createActivityComponent() {
   return new ActivityComponentActivityOnly<RetinaActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // namespace PV
