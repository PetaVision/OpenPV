/*
 * InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "components/InputActivityBuffer.hpp"

namespace PV {

InputLayer::InputLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

InputLayer::~InputLayer() {}

void InputLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *InputLayer::createLayerInput() { return nullptr; }

void InputLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(
            name, "triggerLayerName", nullptr /*correct value*/);
   }
}

void InputLayer::setNontriggerDeltaUpdateTime(double dt) {
   auto *activityComponent = getComponentByType<ActivityComponent>();
   pvAssert(activityComponent);
   auto *activityBuffer = activityComponent->getComponentByType<InputActivityBuffer>();
   pvAssert(activityBuffer);
   auto displayPeriod = activityBuffer->getDisplayPeriod();
   mDeltaUpdateTime   = displayPeriod > 0 ? displayPeriod : DBL_MAX;
}

} // end namespace PV
