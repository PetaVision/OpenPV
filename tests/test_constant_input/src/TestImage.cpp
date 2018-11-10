/*
 * TestImage.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "TestImage.hpp"
#include "TestImageActivityComponent.hpp"

namespace PV {

TestImage::TestImage(const char *name, HyPerCol *hc) { initialize(name, hc); }

TestImage::TestImage() {}

TestImage::~TestImage() {}

int TestImage::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *TestImage::createActivityComponent() {
   return new TestImageActivityComponent(getName(), parent);
}

float TestImage::getConstantVal() const {
   auto *buffer = mActivityComponent->getComponentByType<TestImageActivityBuffer>();
   pvAssert(buffer);
   return buffer->getConstantVal();
}

} // namespace PV
