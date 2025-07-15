/*
 * TestImage.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "TestImage.hpp"
#include "TestImageActivityComponent.hpp"

namespace PV {

TestImage::TestImage(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

TestImage::TestImage() {}

TestImage::~TestImage() {}

void TestImage::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *TestImage::createActivityComponent() {
   return new TestImageActivityComponent(mParamsIO, mCommunicator);
}

float TestImage::getConstantVal() const {
   auto *buffer = mActivityComponent->getComponentByType<TestImageActivityBuffer>();
   pvAssert(buffer);
   return buffer->getConstantVal();
}

} // namespace PV
