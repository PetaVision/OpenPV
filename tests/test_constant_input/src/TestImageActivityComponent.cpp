/*
 * TestImageActivityComponent.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityComponent.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

TestImageActivityComponent::TestImageActivityComponent(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

TestImageActivityComponent::~TestImageActivityComponent() {}

int TestImageActivityComponent::initialize(char const *name, HyPerCol *hc) {
   return ActivityComponent::initialize(name, hc);
}

void TestImageActivityComponent::setObjectType() { mObjectType = "TestImageActivityComponent"; }

ActivityBuffer *TestImageActivityComponent::createActivity() {
   return new TestImageActivityBuffer(getName(), parent);
}

} // namespace PV
