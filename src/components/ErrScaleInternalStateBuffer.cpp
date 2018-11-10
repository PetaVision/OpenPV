/*
 * ErrScaleInternalStateBuffer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ErrScaleInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

ErrScaleInternalStateBuffer::ErrScaleInternalStateBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

ErrScaleInternalStateBuffer::~ErrScaleInternalStateBuffer() {}

int ErrScaleInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = HyPerInternalStateBuffer::initialize(name, hc);
   return status;
}

void ErrScaleInternalStateBuffer::setObjectType() { mObjectType = "ErrScaleInternalStateBuffer"; }

int ErrScaleInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ErrScaleInternalStateBuffer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "errScale", &mErrScale, mErrScale, true /*warnIfAbsent*/);
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
