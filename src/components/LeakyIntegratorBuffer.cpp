/*
 * LeakyIntegratorBuffer.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegratorBuffer.hpp"
#include <cmath>

namespace PV {

LeakyIntegratorBuffer::LeakyIntegratorBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

LeakyIntegratorBuffer::~LeakyIntegratorBuffer() {}

void LeakyIntegratorBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(name, params, comm);
}

void LeakyIntegratorBuffer::setObjectType() { mObjectType = "LeakyIntegratorBuffer"; }

int LeakyIntegratorBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyIntegratorBuffer::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
         "integrationTime",
         &mIntegrationTime,
         mIntegrationTime,
         true /*warnIfAbsent*/);
}

void LeakyIntegratorBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *V          = mBufferData.data();
   float const *GSyn = mGSyn->getBufferData();
   float decayfactor = std::exp(-(float)deltaTime / mIntegrationTime);

   long const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   for (long k = 0; k < numNeuronsAcrossBatch; ++k) {
      float accumulatedGSyn = 0.0f;
      for (int chIdx = 0; chIdx < mNumChannelIndices; ++chIdx) {
         int channel     = mChannelIndices[chIdx];
         float gSynValue = GSyn[channel * numNeuronsAcrossBatch + k];
         accumulatedGSyn += mChannelCoefficients[channel] * gSynValue;
      }
      float value = V[k];
      value *= decayfactor;
      value += accumulatedGSyn;
      V[k] = value;
   }
}

} // namespace PV
