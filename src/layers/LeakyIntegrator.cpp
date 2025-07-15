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

LeakyIntegrator::LeakyIntegrator(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

LeakyIntegrator::LeakyIntegrator() {}

void LeakyIntegrator::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   ANNLayer::initialize(paramsIO, comm);
}

ActivityComponent *LeakyIntegrator::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, LeakyIntegratorBuffer, ANNActivityBuffer>(
         mParamsIO, mCommunicator);
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
