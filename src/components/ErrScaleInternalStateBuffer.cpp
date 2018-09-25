/*
 * ErrScaleInternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "ErrScaleInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

ErrScaleInternalStateBuffer::ErrScaleInternalStateBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

ErrScaleInternalStateBuffer::~ErrScaleInternalStateBuffer() {}

int ErrScaleInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = InternalStateBuffer::initialize(name, hc);
   return status;
}

void ErrScaleInternalStateBuffer::setObjectType() { mObjectType = "ErrScaleInternalStateBuffer"; }

int ErrScaleInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ErrScaleInternalStateBuffer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "errScale", &mErrScale, mErrScale, true /*warnIfAbsent*/);
}

void ErrScaleInternalStateBuffer::updateBuffer(double simTime, double deltaTime) {
   InternalStateBuffer::updateBuffer(simTime, deltaTime);

   float const *gSynHead     = mInputBuffer->getBufferData();
   float *V                  = mBufferData.data();
   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float const *gSynExc      = mInputBuffer->getChannelData(CHANNEL_EXC);
   float const errScale      = mErrScale;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] *= errScale;
   }
}

} // namespace PV
