/*
 * SingleChannelGSynAccumulator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#include "SingleChannelGSynAccumulator.hpp"

namespace PV {

SingleChannelGSynAccumulator::SingleChannelGSynAccumulator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

SingleChannelGSynAccumulator::~SingleChannelGSynAccumulator() {}

void SingleChannelGSynAccumulator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   GSynAccumulator::initialize(name, params, comm);
}

void SingleChannelGSynAccumulator::setObjectType() { mObjectType = "SingleChannelGSynAccumulator"; }

void SingleChannelGSynAccumulator::initializeChannelCoefficients() {
   mChannelCoefficients = {1.0f};
}

void SingleChannelGSynAccumulator::ioParam_channelIndices(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void SingleChannelGSynAccumulator::ioParam_channelCoefficients(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
      WarnLog().printf(
            "%s does not use the channelCoefficients array parameter.\n", getDescription_c());
   }
}

Response::Status SingleChannelGSynAccumulator::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return GSynAccumulator::communicateInitInfo(message);
}

void SingleChannelGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   float const *gSynExc  = mLayerInput->getChannelData(CHANNEL_EXC);
   float *bufferData     = mBufferData.data();
   int numNeurons        = getBufferSizeAcrossBatch();
   int numChannels       = (int)mChannelCoefficients.size();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      bufferData[kIndex] = gSynExc[kIndex];
   }
}

} // namespace PV
