/*
 * LIFLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "LIFLayerInputBuffer.hpp"

namespace PV {

LIFLayerInputBuffer::LIFLayerInputBuffer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

LIFLayerInputBuffer::~LIFLayerInputBuffer() {}

void LIFLayerInputBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   LayerInputBuffer::initialize(name, params, comm);
}

int LIFLayerInputBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerInputBuffer::ioParamsFillGroup(ioFlag);
   ioParam_tauE(ioFlag);
   ioParam_tauI(ioFlag);
   ioParam_tauIB(ioFlag);
   return status;
}

void LIFLayerInputBuffer::ioParam_tauE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauE", &mTauE, mTauE);
}

void LIFLayerInputBuffer::ioParam_tauI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauI", &mTauI, mTauI);
}

void LIFLayerInputBuffer::ioParam_tauIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauIB", &mTauIB, mTauIB);
}

void LIFLayerInputBuffer::setObjectType() { mObjectType = "LIFLayerInputBuffer"; }

void LIFLayerInputBuffer::initChannelTimeConstants() {
   LayerInputBuffer::initChannelTimeConstants();
   pvAssert(mNumChannels >= 3 && mChannelTimeConstants.size() >= (std::size_t)3);
   mChannelTimeConstants[CHANNEL_EXC]  = mTauE;
   mChannelTimeConstants[CHANNEL_INH]  = mTauI;
   mChannelTimeConstants[CHANNEL_INHB] = mTauIB;
}

void LIFLayerInputBuffer::recvUnitInput(float *recvBuffer) {
   for (auto &d : mDeliverySources) {
      if (d == nullptr or d->getChannelCode() != CHANNEL_GAP) {
         continue;
      }
      d->deliverUnitInput(recvBuffer);
   }
}

} // namespace PV
