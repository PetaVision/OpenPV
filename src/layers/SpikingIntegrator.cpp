/*
 * SpikingIntegrator.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: twatkins
 */

#include "SpikingIntegrator.hpp"
#include "components/SpikingActivityComponent.hpp"

namespace PV {

SpikingIntegrator::SpikingIntegrator(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

SpikingIntegrator::SpikingIntegrator() {}

SpikingIntegrator::~SpikingIntegrator() {}

void SpikingIntegrator::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *SpikingIntegrator::createActivityComponent() {
   return new SpikingActivityComponent(name, parameters(), mCommunicator);
}

} /* namespace PV */
