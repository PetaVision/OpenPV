/*
 * TestImage.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "TestImage.hpp"
#include "TestImageActivityComponent.hpp"

namespace PV {

TestImage::TestImage(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

TestImage::TestImage() {}

TestImage::~TestImage() {}

void TestImage::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *TestImage::createActivityComponent() {
   return new TestImageActivityComponent(getName(), parameters(), mCommunicator);
}

float TestImage::getConstantVal() const {
   auto *buffer = mActivityComponent->getComponentByType<TestImageActivityBuffer>();
   pvAssert(buffer);
   return buffer->getConstantVal();
}

} // namespace PV
