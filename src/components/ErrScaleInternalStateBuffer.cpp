/*
 * ErrScaleInternalStateBuffer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ErrScaleInternalStateBuffer.hpp"

namespace PV {

ErrScaleInternalStateBuffer::ErrScaleInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ErrScaleInternalStateBuffer::~ErrScaleInternalStateBuffer() {}

void ErrScaleInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(name, params, comm);
}

void ErrScaleInternalStateBuffer::setObjectType() { mObjectType = "ErrScaleInternalStateBuffer"; }

int ErrScaleInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ErrScaleInternalStateBuffer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, getName(), "errScale", &mErrScale, mErrScale, true /*warnIfAbsent*/);
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
