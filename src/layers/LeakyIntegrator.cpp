/*
 * LeakyIntegrator.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegrator.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/LeakyIntegratorBuffer.hpp"

namespace PV {

LeakyIntegrator::LeakyIntegrator(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

LeakyIntegrator::LeakyIntegrator() { initialize_base(); }

int LeakyIntegrator::initialize_base() {
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

void LeakyIntegrator::initialize(const char *name, PVParams *params, Communicator *comm) {
   ANNLayer::initialize(name, params, comm);
}

ActivityComponent *LeakyIntegrator::createActivityComponent() {
   return new ActivityComponentWithInternalState<LeakyIntegratorBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
