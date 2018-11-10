/*
 * SquaredInternalStateBuffer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "SquaredInternalStateBuffer.hpp"

namespace PV {

SquaredInternalStateBuffer::SquaredInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

SquaredInternalStateBuffer::~SquaredInternalStateBuffer() {}

void SquaredInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   GSynInternalStateBuffer::initialize(name, params, comm);
}

void SquaredInternalStateBuffer::setObjectType() { mObjectType = "SquaredInternalStateBuffer"; }

void SquaredInternalStateBuffer::requireInputChannels() {
   mLayerInput->requireChannel(CHANNEL_EXC);
}

void SquaredInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynHead = mLayerInput->getBufferData();
   float *V              = mBufferData.data();

   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float const *gSynExc      = mLayerInput->getChannelData(CHANNEL_EXC);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      float g = gSynExc[k];
      V[k]    = g * g;
   }
}

} // namespace PV
