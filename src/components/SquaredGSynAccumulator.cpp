/*
 * SquaredGSynAccumulator.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "SquaredGSynAccumulator.hpp"

namespace PV {

SquaredGSynAccumulator::SquaredGSynAccumulator(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

SquaredGSynAccumulator::~SquaredGSynAccumulator() {}

void SquaredGSynAccumulator::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   SingleChannelGSynAccumulator::initialize(paramsIO, comm);
}

void SquaredGSynAccumulator::setObjectType() { mObjectType = "SquaredGSynAccumulator"; }

void SquaredGSynAccumulator::initializeChannelCoefficients() {
   mChannelCoefficients.resize(1);
   mChannelCoefficients[0] = 1.0f;
}

void SquaredGSynAccumulator::updateBufferCPU(double simTime, double deltaTime) {
   float const *gSynExc = mLayerInput->getChannelData(CHANNEL_EXC);
   float *bufferData    = mBufferData.data();
   int numNeurons       = getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons; kIndex++) {
      float gSyn         = gSynExc[kIndex];
      bufferData[kIndex] = gSyn * gSyn;
   }
}

} // namespace PV
