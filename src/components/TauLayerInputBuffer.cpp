/*
 * TauLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "TauLayerInputBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

TauLayerInputBuffer::TauLayerInputBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

TauLayerInputBuffer::~TauLayerInputBuffer() {}

int TauLayerInputBuffer::initialize(char const *name, HyPerCol *hc) {
   return LayerInputBuffer::initialize(name, hc);
}

int TauLayerInputBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerInputBuffer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   return status;
}

void TauLayerInputBuffer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "timeConstantTau", &mTimeConstantTau, mTimeConstantTau);
}

void TauLayerInputBuffer::setObjectType() { mObjectType = "TauLayerInputBuffer"; }

void TauLayerInputBuffer::initChannelTimeConstants() {
   for (auto &c : mChannelTimeConstants) {
      c = mTimeConstantTau;
   }
}

} // namespace PV
