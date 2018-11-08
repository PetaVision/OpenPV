/*
 * LeakyIntegratorBuffer.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegratorBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

LeakyIntegratorBuffer::LeakyIntegratorBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

LeakyIntegratorBuffer::~LeakyIntegratorBuffer() {}

void LeakyIntegratorBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   GSynInternalStateBuffer::initialize(name, params, comm);
}

void LeakyIntegratorBuffer::setObjectType() { mObjectType = "LeakyIntegratorBuffer"; }

int LeakyIntegratorBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = GSynInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyIntegratorBuffer::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "integrationTime",
         &mIntegrationTime,
         mIntegrationTime,
         true /*warnIfAbsent*/);
}

void LeakyIntegratorBuffer::requireInputChannels() { mLayerInput->requireChannel(CHANNEL_EXC); }

void LeakyIntegratorBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float *V             = mBufferData.data();

   float decayfactor                 = std::exp(-(float)deltaTime / mIntegrationTime);
   float const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] *= decayfactor;
      V[k] += gSynExc[k];
   }
   if (mLayerInput->getNumChannels() > 1) {
      float const *gSynInh = mLayerInput->getChannelData(CHANNEL_INH);
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] -= gSynInh[k];
      }
   }
}

} // namespace PV
