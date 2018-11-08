/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ConstantLayer::ConstantLayer() {}

ConstantLayer::~ConstantLayer() {}

void ConstantLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   // HyPerLayer default for writeStep is 1.0, but
   // writeStep = -1 (never write) is a better default for ConstantLayer.
   writeStep = -1;

   HyPerLayer::initialize(name, params, comm);
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
