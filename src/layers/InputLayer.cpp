/*
 * InputLayer.cpp
 */

#include "InputLayer.hpp"
#include "components/InputActivityBuffer.hpp"

namespace PV {

InputLayer::InputLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

InputLayer::~InputLayer() {}

int InputLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
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
