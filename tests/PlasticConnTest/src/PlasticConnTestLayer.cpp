/*
 * PlasticConnTestLayer.cpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#include "PlasticConnTestLayer.hpp"

#include "PlasticConnTestActivityBuffer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"

namespace PV {

PlasticConnTestLayer::PlasticConnTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void PlasticConnTestLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *PlasticConnTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PlasticConnTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
