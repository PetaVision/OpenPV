/*
 * LeakyInternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "LeakyInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

LeakyInternalStateBuffer::LeakyInternalStateBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

LeakyInternalStateBuffer::~LeakyInternalStateBuffer() {}

int LeakyInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = InternalStateBuffer::initialize(name, hc);
   return status;
}

void LeakyInternalStateBuffer::setObjectType() { mObjectType = "LeakyInternalStateBuffer"; }

int LeakyInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyInternalStateBuffer::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "integrationTime",
         &mIntegrationTime,
         mIntegrationTime,
         true /*warnIfAbsent*/);
}

void LeakyInternalStateBuffer::updateBuffer(double simTime, double deltaTime) {
   float const *gSyn = mInputBuffer->getChannelData(CHANNEL_EXC);
   float *V          = mBufferData.data();

   float decayfactor                 = std::exp(-(float)deltaTime / mIntegrationTime);
   float const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] *= decayfactor;
      V[k] += gSyn[k];
   }
   if (mInputBuffer->getNumChannels() > 1) {
      float const *gSynInh = mInputBuffer->getChannelData(CHANNEL_INH);
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] -= gSynInh[k];
      }
   }
}

} // namespace PV
