/*
 * TestImage.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "TestImage.hpp"
#include "TestImageActivityComponent.hpp"

namespace PV {

TestImage::TestImage(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

TestImage::TestImage() {}

TestImage::~TestImage() {}

void TestImage::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *TestImage::createActivityComponent() {
   return new TestImageActivityComponent(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

float TestImage::getConstantVal() const {
   auto *buffer = mActivityComponent->getComponentByType<TestImageActivityBuffer>();
   pvAssert(buffer);
   return buffer->getConstantVal();
}

} // namespace PV
