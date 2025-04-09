/*
 * InputVolumeActivityComponent.cpp
 */

#include "InputVolumeActivityComponent.hpp"
#include "components/InputVolumeActivityBuffer.hpp"

namespace PV {

InputVolumeActivityComponent::InputVolumeActivityComponent(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ActivityBuffer *InputVolumeActivityComponent::createActivity() {
   return new InputVolumeActivityBuffer(getName(), parameters(), mCommunicator);
}

void InputVolumeActivityComponent::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   ActivityComponent::initialize(name, params, comm);
}

void InputVolumeActivityComponent::setObjectType() { mObjectType = "InputVolumeActivityComponent"; }

} // namespace PV
