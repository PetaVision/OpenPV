/*
 * InputRegionActivityComponent.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionActivityComponent.hpp"
#include "columns/HyPerCol.hpp"
#include "components/InputRegionActivityBuffer.hpp"

// InputRegionActivityComponent clones an InputLayer's InputRegion buffer
// as its activity.
namespace PV {
InputRegionActivityComponent::InputRegionActivityComponent() {}

InputRegionActivityComponent::InputRegionActivityComponent(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

InputRegionActivityComponent::~InputRegionActivityComponent() {}

int InputRegionActivityComponent::initialize(const char *name, HyPerCol *hc) {
   int status_init = ActivityComponent::initialize(name, hc);
   return status_init;
}

void InputRegionActivityComponent::setObjectType() { mObjectType = "InputRegionActivityComponent"; }

ActivityBuffer *InputRegionActivityComponent::createActivity() {
   return new InputRegionActivityBuffer(getName(), parent);
}

Response::Status InputRegionActivityComponent::updateActivity(double simTime, double deltaTime) {
   return Response::NO_ACTION;
}

void InputRegionActivityComponent::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mUpdateGpu = false;
      parameters()->handleUnnecessaryParameter(name, "updateGpu", mUpdateGpu);
   }
}

} // end namespace PV
