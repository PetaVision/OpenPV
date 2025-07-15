/*
 * InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "components/InputLayerUpdateController.hpp"

namespace PV {

InputLayer::InputLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

InputLayer::~InputLayer() {}

void InputLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

LayerUpdateController *InputLayer::createLayerUpdateController() {
   return new InputLayerUpdateController(mParamsIO, mCommunicator);
}

LayerInputBuffer *InputLayer::createLayerInput() { return nullptr; }

} // end namespace PV
