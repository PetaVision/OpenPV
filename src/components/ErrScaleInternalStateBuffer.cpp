/*
 * ErrScaleInternalStateBuffer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ErrScaleInternalStateBuffer.hpp"

namespace PV {

ErrScaleInternalStateBuffer::ErrScaleInternalStateBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ErrScaleInternalStateBuffer::~ErrScaleInternalStateBuffer() {}

void ErrScaleInternalStateBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(params, defaults, comm);
}

void ErrScaleInternalStateBuffer::setObjectType() { mObjectType = "ErrScaleInternalStateBuffer"; }

int ErrScaleInternalStateBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_errScale(ioSwitch);
   return status;
}

void ErrScaleInternalStateBuffer::ioParam_errScale(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "errScale", &mErrScale);
}

void ErrScaleInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   HyPerInternalStateBuffer::updateBufferCPU(simTime, deltaTime);

   float *V                  = mBufferData.data();
   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float const errScale      = mErrScale;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] *= errScale;
   }
}

} // namespace PV
