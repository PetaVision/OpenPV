/*
 * InputRegionActivityComponent.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionActivityComponent.hpp"
#include "components/InputRegionActivityBuffer.hpp"

// InputRegionActivityComponent clones an InputLayer's InputRegion buffer
// as its activity.
namespace PV {
InputRegionActivityComponent::InputRegionActivityComponent() {}

InputRegionActivityComponent::InputRegionActivityComponent(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputRegionActivityComponent::~InputRegionActivityComponent() {}

void InputRegionActivityComponent::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityComponent::initialize(name, params, comm);
}

void InputRegionActivityComponent::setObjectType() { mObjectType = "InputRegionActivityComponent"; }

ActivityBuffer *InputRegionActivityComponent::createActivity() {
   return new InputRegionActivityBuffer(getName(), parameters(), mCommunicator);
}

void InputRegionActivityComponent::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mUpdateGpu = false;
      parameters()->handleUnnecessaryParameter(name, "updateGpu", mUpdateGpu);
   }
}

Response::Status InputRegionActivityComponent::updateActivity(double simTime, double deltaTime) {
   return Response::NO_ACTION;
}

} // end namespace PV
