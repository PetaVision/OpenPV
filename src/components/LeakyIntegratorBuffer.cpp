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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

LeakyIntegratorBuffer::~LeakyIntegratorBuffer() {}

void LeakyIntegratorBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(params, defaults, comm);
}

void LeakyIntegratorBuffer::setObjectType() { mObjectType = "LeakyIntegratorBuffer"; }

int LeakyIntegratorBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_integrationTime(ioSwitch);
   return status;
}

void LeakyIntegratorBuffer::ioParam_integrationTime(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "integrationTime", &mIntegrationTime);
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
