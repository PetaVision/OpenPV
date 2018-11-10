/*
 * PtwiseQuotientInternalStateBuffer.cpp
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
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
   int status = GSynInternalStateBuffer::initialize(name, hc);
   return status;
}

void PtwiseQuotientInternalStateBuffer::setObjectType() {
   mObjectType = "PtwiseQuotientInternalStateBuffer";
}

void PtwiseQuotientInternalStateBuffer::requireInputChannels() {
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);
}

void PtwiseQuotientInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc      = mLayerInput->getChannelData(CHANNEL_EXC);
   float const *gSynInh      = mLayerInput->getChannelData(CHANNEL_INH);
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
