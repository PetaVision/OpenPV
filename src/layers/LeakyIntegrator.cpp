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

LeakyIntegrator::LeakyIntegrator(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

LeakyIntegrator::LeakyIntegrator() { initialize_base(); }

int LeakyIntegrator::initialize_base() {
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

int LeakyIntegrator::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   return status;
}

ActivityComponent *LeakyIntegrator::createActivityComponent() {
   return new ActivityComponentWithInternalState<LeakyIntegratorBuffer, ANNActivityBuffer>(
         getName(), parent);
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
