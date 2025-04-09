/*
 * InputVolumeLayer.cpp
 */

#include "InputVolumeLayer.hpp"
#include "components/InputVolumeActivityComponent.hpp"
#include "components/InputVolumeLayerUpdateController.hpp"

namespace PV {

InputVolumeLayer::InputVolumeLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

InputVolumeLayer::~InputVolumeLayer() {}

void InputVolumeLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *InputVolumeLayer::createActivityComponent() {
   return new InputVolumeActivityComponent(getName(), parameters(), mCommunicator);
}

LayerUpdateController *InputVolumeLayer::createLayerUpdateController() {
   return new InputVolumeLayerUpdateController(getName(), parameters(), mCommunicator);
}

LayerInputBuffer *InputVolumeLayer::createLayerInput() { return nullptr; }

} // end namespace PV
