/*
 * TestImageActivityComponent.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityComponent.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

TestImageActivityComponent::TestImageActivityComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

TestImageActivityComponent::~TestImageActivityComponent() {}

void TestImageActivityComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   ActivityComponent::initialize(name, params, comm);
}

void TestImageActivityComponent::setObjectType() { mObjectType = "TestImageActivityComponent"; }

ActivityBuffer *TestImageActivityComponent::createActivity() {
   return new TestImageActivityBuffer(getName(), parameters(), mCommunicator);
}

} // namespace PV
