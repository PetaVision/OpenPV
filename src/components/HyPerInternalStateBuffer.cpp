/*
 * HyPerInternalStateBuffer.cpp
 *
 *  Created on: Oct 12, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include "HyPerInternalStateBuffer.hpp"

namespace PV {

HyPerInternalStateBuffer::HyPerInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

HyPerInternalStateBuffer::~HyPerInternalStateBuffer() {}

void HyPerInternalStateBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   GSynInternalStateBuffer::initialize(name, params, comm);
}

void HyPerInternalStateBuffer::setObjectType() { mObjectType = "HyPerInternalStateBuffer"; }

void HyPerInternalStateBuffer::requireInputChannels() { mLayerInput->requireChannel(CHANNEL_EXC); }

void HyPerInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *V = getReadWritePointer();
   if (V == nullptr) {
      WarnLog().printf(
            "%s is not updateable. updateBuffer called with t=%f, dt=%f.\n",
            getDescription(),
            simTime,
            deltaTime);
      return;
   }
   int const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   pvAssert(numNeuronsAcrossBatch == mLayerInput->getBufferSizeAcrossBatch());

   int const numChannels = mLayerInput->getNumChannels();
   pvAssert(numChannels >= 1); // communicateInitInfo called requireChannel with channel 0.
   if (numChannels == 1) {
      float const *excitatoryInput = mLayerInput->getChannelData(CHANNEL_EXC);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] = excitatoryInput[k];
      }
   }
   else {
      float const *excitatoryInput = mLayerInput->getChannelData(CHANNEL_EXC);
      float const *inhibitoryInput = mLayerInput->getChannelData(CHANNEL_INH);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] = excitatoryInput[k] - inhibitoryInput[k];
      }
   }
}

} // namespace PV
