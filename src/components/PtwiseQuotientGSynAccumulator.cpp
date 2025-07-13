/*
 * PtwiseQuotientGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "PtwiseQuotientGSynAccumulator.hpp"

namespace PV {

PtwiseQuotientGSynAccumulator::PtwiseQuotientGSynAccumulator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PtwiseQuotientGSynAccumulator::~PtwiseQuotientGSynAccumulator() {}

void PtwiseQuotientGSynAccumulator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   GSynAccumulator::initialize(params, defaults, comm);
}

void PtwiseQuotientGSynAccumulator::setObjectType() {
   mObjectType = "PtwiseQuotientGSynAccumulator";
}

void PtwiseQuotientGSynAccumulator::ioParam_channelIndices(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void PtwiseQuotientGSynAccumulator::ioParam_channelCoefficients(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelCoefficients")) {
      WarnLog().printf(
            "%s does not use the channelCoefficients array parameter.\n", getDescription_c());
   }
}

Response::Status PtwiseQuotientGSynAccumulator::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return GSynAccumulator::communicateInitInfo(message);
}

void PtwiseQuotientGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float const *gSynInh = mLayerInput->getChannelData(CHANNEL_INH);
   float *bufferData    = mBufferData.data();
   int numNeurons       = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      bufferData[kIndex] = gSynExc[kIndex] / gSynInh[kIndex];
   }
}

} // namespace PV
