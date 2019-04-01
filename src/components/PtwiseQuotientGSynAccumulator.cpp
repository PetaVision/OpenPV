/*
 * PtwiseQuotientGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "PtwiseQuotientGSynAccumulator.hpp"

namespace PV {

PtwiseQuotientGSynAccumulator::PtwiseQuotientGSynAccumulator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PtwiseQuotientGSynAccumulator::~PtwiseQuotientGSynAccumulator() {}

void PtwiseQuotientGSynAccumulator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   GSynAccumulator::initialize(name, params, comm);
}

void PtwiseQuotientGSynAccumulator::setObjectType() {
   mObjectType = "PtwiseQuotientGSynAccumulator";
}

void PtwiseQuotientGSynAccumulator::ioParam_channelIndices(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void PtwiseQuotientGSynAccumulator::ioParam_channelCoefficients(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
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
