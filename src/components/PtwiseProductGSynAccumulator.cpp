/*
 * PtwiseProductGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "PtwiseProductGSynAccumulator.hpp"

namespace PV {

PtwiseProductGSynAccumulator::PtwiseProductGSynAccumulator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PtwiseProductGSynAccumulator::~PtwiseProductGSynAccumulator() {}

void PtwiseProductGSynAccumulator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   GSynAccumulator::initialize(name, params, comm);
}

void PtwiseProductGSynAccumulator::setObjectType() { mObjectType = "PtwiseProductGSynAccumulator"; }

void PtwiseProductGSynAccumulator::ioParam_channelIndices(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
      WarnLog().printf("%s does not use the channelIndices array parameter.\n", getDescription_c());
   }
}

void PtwiseProductGSynAccumulator::ioParam_channelCoefficients(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->arrayPresent(getName(), "channelIndices")) {
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
   long numNeurons      = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (long index = 0; index < numNeurons; index++) {
      bufferData[index] = gSynExc[index] * gSynInh[index];
   }
}

} // namespace PV
