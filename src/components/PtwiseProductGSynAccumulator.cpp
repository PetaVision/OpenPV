/*
 * PtwiseProductGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "PtwiseProductGSynAccumulator.hpp"

namespace PV {

PtwiseProductGSynAccumulator::PtwiseProductGSynAccumulator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PtwiseProductGSynAccumulator::~PtwiseProductGSynAccumulator() {}

void PtwiseProductGSynAccumulator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   GSynAccumulator::initialize(params, defaults, comm);
}

void PtwiseProductGSynAccumulator::setObjectType() { mObjectType = "PtwiseProductGSynAccumulator"; }

void PtwiseProductGSynAccumulator::ioParam_channelIndices(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void PtwiseProductGSynAccumulator::ioParam_channelCoefficients(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read and mParamsIO->isPresent("channelCoefficients")) {
      WarnLog().printf(
            "%s does not use the channelCoefficients array parameter.\n", getDescription_c());
   }
}

Response::Status PtwiseProductGSynAccumulator::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return GSynAccumulator::communicateInitInfo(message);
}

void PtwiseProductGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float const *gSynInh = mLayerInput->getChannelData(CHANNEL_INH);
   float *bufferData    = mBufferData.data();
   int numNeurons       = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      bufferData[kIndex] = gSynExc[kIndex] * gSynInh[kIndex];
   }
}

} // namespace PV
