/*
 * TestImageActivityComponent.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityComponent.hpp"

namespace PV {

TestImageActivityComponent::TestImageActivityComponent(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

TestImageActivityComponent::~TestImageActivityComponent() {}

void TestImageActivityComponent::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityComponent::initialize(paramsIO, comm);
}

void TestImageActivityComponent::setObjectType() { mObjectType = "TestImageActivityComponent"; }

ActivityBuffer *TestImageActivityComponent::createActivity() {
   return new TestImageActivityBuffer(mParamsIO, mCommunicator);
}

} // namespace PV
