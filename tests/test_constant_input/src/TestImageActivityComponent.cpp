/*
 * TestImageActivityComponent.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityComponent.hpp"

namespace PV {

TestImageActivityComponent::TestImageActivityComponent(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

TestImageActivityComponent::~TestImageActivityComponent() {}

void TestImageActivityComponent::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ActivityComponent::initialize(params, defaults, comm);
}

void TestImageActivityComponent::setObjectType() { mObjectType = "TestImageActivityComponent"; }

ActivityBuffer *TestImageActivityComponent::createActivity() {
   return new TestImageActivityBuffer(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
