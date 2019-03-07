/*
 * SquaredGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "SquaredGSynAccumulator.hpp"

namespace PV {

SquaredGSynAccumulator::SquaredGSynAccumulator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

SquaredGSynAccumulator::~SquaredGSynAccumulator() {}

void SquaredGSynAccumulator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   SingleChannelGSynAccumulator::initialize(name, params, comm);
}

void SquaredGSynAccumulator::setObjectType() { mObjectType = "SquaredGSynAccumulator"; }

void SquaredGSynAccumulator::initializeChannelCoefficients() { mChannelCoefficients = {1.0f}; }

void SquaredGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   float const *gSynExc  = mLayerInput->getChannelData(CHANNEL_EXC);
   float *bufferData     = mBufferData.data();
   int numNeurons        = getBufferSizeAcrossBatch();
   int numChannels       = (int)mChannelCoefficients.size();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      float gSyn         = gSynExc[kIndex];
      bufferData[kIndex] = gSyn * gSyn;
   }
}

} // namespace PV
