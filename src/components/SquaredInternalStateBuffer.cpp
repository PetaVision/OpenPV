/*
 * SquaredInternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "SquaredInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

SquaredInternalStateBuffer::SquaredInternalStateBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

SquaredInternalStateBuffer::~SquaredInternalStateBuffer() {}

int SquaredInternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = InternalStateBuffer::initialize(name, hc);
   return status;
}

void SquaredInternalStateBuffer::setObjectType() { mObjectType = "SquaredInternalStateBuffer"; }

void SquaredInternalStateBuffer::updateBuffer(double simTime, double deltaTime) {
   float const *gSynHead = mInputBuffer->getBufferData();
   float *V              = mBufferData.data();

   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float const *gSynExc      = mInputBuffer->getChannelData(CHANNEL_EXC);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      float g = gSynExc[k];
      V[k]    = g * g;
   }
}

} // namespace PV
