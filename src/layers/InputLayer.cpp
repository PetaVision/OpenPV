/*
 * InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "components/InputLayerUpdateController.hpp"

namespace PV {

InputLayer::InputLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

InputLayer::~InputLayer() {}

void InputLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerUpdateController *InputLayer::createLayerUpdateController() {
   return new InputLayerUpdateController(getName(), parameters(), mCommunicator);
}

LayerInputBuffer *InputLayer::createLayerInput() { return nullptr; }

} // end namespace PV
