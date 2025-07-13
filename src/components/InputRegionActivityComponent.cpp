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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputRegionActivityComponent::~InputRegionActivityComponent() {}

void InputRegionActivityComponent::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ActivityComponent::initialize(params, defaults, comm);
}

void InputRegionActivityComponent::setObjectType() { mObjectType = "InputRegionActivityComponent"; }

ActivityBuffer *InputRegionActivityComponent::createActivity() {
   return new InputRegionActivityBuffer(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

void InputRegionActivityComponent::ioParam_updateGpu(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mUpdateGpu = false;
      mParamsIO->handleUnnecessaryParameter("updateGpu", mUpdateGpu);
   }
}

Response::Status InputRegionActivityComponent::updateActivity(double simTime, double deltaTime) {
   return Response::NO_ACTION;
}

} // end namespace PV
