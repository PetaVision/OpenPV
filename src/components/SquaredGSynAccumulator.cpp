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

void SquaredGSynAccumulator::initializeChannelCoefficients() {
   mChannelCoefficients.resize(1);
   mChannelCoefficients[0] = 1.0f;
}

void SquaredGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float *bufferData    = mBufferData.data();
   long numNeurons      = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (long neuron = 0; neuron < numNeurons; neuron++) {
      float gSyn         = gSynExc[neuron];
      bufferData[neuron] = gSyn * gSyn;
   }
}

} // namespace PV
