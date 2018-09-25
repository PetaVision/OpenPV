/*
 * PtwiseQuotientInternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "PtwiseQuotientInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PtwiseQuotientInternalStateBuffer::PtwiseQuotientInternalStateBuffer(
      char const *name,
      HyPerCol *hc) {
   initialize(name, hc);
}

PtwiseQuotientInternalStateBuffer::~PtwiseQuotientInternalStateBuffer() {}

int PtwiseQuotientInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = InternalStateBuffer::initialize(name, hc);
   return status;
}

void PtwiseQuotientInternalStateBuffer::setObjectType() {
   mObjectType = "PtwiseQuotientInternalStateBuffer";
}

void PtwiseQuotientInternalStateBuffer::updateBuffer(double simTime, double deltaTime) {
   float const *gSynExc      = mInputBuffer->getChannelData(CHANNEL_EXC);
   float const *gSynInh      = mInputBuffer->getChannelData(CHANNEL_INH);
   float *V                  = mBufferData.data();
   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] = gSynExc[k] / gSynInh[k];
   }
}

} // namespace PV
