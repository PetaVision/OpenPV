/*
 * PtwiseProductInternalStateBuffer.cpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseProductInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PtwiseProductInternalStateBuffer::PtwiseProductInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

PtwiseProductInternalStateBuffer::~PtwiseProductInternalStateBuffer() {}

void PtwiseProductInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   GSynInternalStateBuffer::initialize(name, params, comm);
}

void PtwiseProductInternalStateBuffer::setObjectType() {
   mObjectType = "PtwiseProductInternalStateBuffer";
}

void PtwiseProductInternalStateBuffer::requireInputChannels() {
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);
}

void PtwiseProductInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc      = mLayerInput->getChannelData(CHANNEL_EXC);
   float const *gSynInh      = mLayerInput->getChannelData(CHANNEL_INH);
   float *V                  = mBufferData.data();
   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] = gSynExc[k] * gSynInh[k];
   }
}

} // namespace PV
