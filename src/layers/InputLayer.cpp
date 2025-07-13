/*
 * InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "components/InputLayerUpdateController.hpp"

namespace PV {

InputLayer::InputLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputLayer::~InputLayer() {}

void InputLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

LayerUpdateController *InputLayer::createLayerUpdateController() {
   return new InputLayerUpdateController(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LayerInputBuffer *InputLayer::createLayerInput() { return nullptr; }

} // end namespace PV
