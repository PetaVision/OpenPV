/*
 * LeakyIntegrator.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegrator.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/LeakyIntegratorBuffer.hpp"

namespace PV {

LeakyIntegrator::LeakyIntegrator(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

LeakyIntegrator::LeakyIntegrator() {}

void LeakyIntegrator::initialize(const char *name, PVParams *params, Communicator *comm) {
   ANNLayer::initialize(name, params, comm);
}

ActivityComponent *LeakyIntegrator::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, LeakyIntegratorBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
