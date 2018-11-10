/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ConstantLayer::ConstantLayer() {}

ConstantLayer::~ConstantLayer() {}

int ConstantLayer::initialize(const char *name, HyPerCol *hc) {
   // HyPerLayer default for writeStep is 1.0, but
   // writeStep = -1 (never write) is a better default for ConstantLayer.
   writeStep = -1;

   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void ConstantLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   // This layer is a never a trigger layer, so set to null
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", nullptr);
   }
}

bool ConstantLayer::needUpdate(double simTime, double dt) const { return false; }

} /* namespace PV */
