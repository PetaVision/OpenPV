/*
 * CPTestInputLayer.cpp
 */

#include "CPTestInputLayer.hpp"
#include "CPTestInputInternalStateBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

CPTestInputLayer::CPTestInputLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

CPTestInputLayer::~CPTestInputLayer() {}

int CPTestInputLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *CPTestInputLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CPTestInputInternalStateBuffer,
                                                 HyPerActivityBuffer>(getName(), parent);
}

} // end namespace PV
