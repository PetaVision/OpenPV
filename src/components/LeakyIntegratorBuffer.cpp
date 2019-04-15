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
         name,
         "integrationTime",
         &mIntegrationTime,
         mIntegrationTime,
         true /*warnIfAbsent*/);
}

void LeakyIntegratorBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSyn = mAccumulatedGSyn->getBufferData();
   float *V          = mBufferData.data();

   float decayfactor                 = std::exp(-(float)deltaTime / mIntegrationTime);
   float const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] *= decayfactor;
      V[k] += gSyn[k];
   }
}

} // namespace PV
