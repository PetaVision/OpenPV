/*
 * MPITestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "MPITestLayer.hpp"

#include "MPITestActivityBuffer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"

namespace PV {

MPITestLayer::MPITestLayer(const char *name, PVParams *params, Communicator const *comm)
      : HyPerLayer() {
   // MPITestLayer has no member variables to initialize in initialize_base()
   initialize(name, params, comm);
}

void MPITestLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *MPITestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<MPITestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
