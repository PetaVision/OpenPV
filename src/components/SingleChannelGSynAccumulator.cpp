/*
 * SingleChannelGSynAccumulator.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#include "SingleChannelGSynAccumulator.hpp"

namespace PV {

SingleChannelGSynAccumulator::SingleChannelGSynAccumulator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

SingleChannelGSynAccumulator::~SingleChannelGSynAccumulator() {}

void SingleChannelGSynAccumulator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   GSynAccumulator::initialize(params, defaults, comm);
}

void SingleChannelGSynAccumulator::setObjectType() { mObjectType = "SingleChannelGSynAccumulator"; }

void SingleChannelGSynAccumulator::initializeChannelCoefficients() {
   mChannelCoefficients.resize(1);
   mChannelCoefficients[0] = 1.0f;
}

void SingleChannelGSynAccumulator::ioParam_channelIndices(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void SingleChannelGSynAccumulator::ioParam_channelCoefficients(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelCoefficients")) {
      WarnLog().printf(
            "%s does not use the channelCoefficients array parameter.\n", getDescription_c());
   }
}

Response::Status SingleChannelGSynAccumulator::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return GSynAccumulator::communicateInitInfo(message);
}

void SingleChannelGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float *bufferData    = mBufferData.data();
   int numNeurons       = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      bufferData[kIndex] = gSynExc[kIndex];
   }
}

} // namespace PV
