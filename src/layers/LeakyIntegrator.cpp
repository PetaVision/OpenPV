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

LeakyIntegrator::LeakyIntegrator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

LeakyIntegrator::LeakyIntegrator() {}

void LeakyIntegrator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ANNLayer::initialize(params, defaults, comm);
}

ActivityComponent *LeakyIntegrator::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, LeakyIntegratorBuffer, ANNActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
