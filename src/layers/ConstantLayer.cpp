/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer() { initialize_base(); }

ConstantLayer::ConstantLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ConstantLayer::~ConstantLayer() {}

int ConstantLayer::initialize_base() {
   writeStep = -1; // HyPerLayer default for writeStep is 1.0, but -1 (never write) is a better
   // default for ConstantLayer
   return PV_SUCCESS;
}

int ConstantLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void ConstantLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   // This layer is a never a trigger layer, so set to null
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", nullptr);
   }
}

int ConstantLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = HyPerLayer::communicateInitInfo(message);
   return status;
}

// bool ConstantLayer::checkIfUpdateNeeded() {
bool ConstantLayer::needUpdate(double time, double dt) {
   // Only update on initialization
   assert(time >= parent->getStartTime());
   if (time == parent->getStartTime()) {
      return true;
   }
   else {
      return false;
   }
}

} /* namespace PV */
