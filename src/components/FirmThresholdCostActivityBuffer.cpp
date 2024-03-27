/*
 * FirmThresholdCostActivityBuffer.cpp
 *
 *  Created on: Apr 2, 2019
 *      Author: pschultz
 */

#include "FirmThresholdCostActivityBuffer.hpp"
#include <cmath>

namespace PV {

FirmThresholdCostActivityBuffer::FirmThresholdCostActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FirmThresholdCostActivityBuffer::~FirmThresholdCostActivityBuffer() {}

void FirmThresholdCostActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

void FirmThresholdCostActivityBuffer::setObjectType() {
   mObjectType = "FirmThresholdCostActivityBuffer";
}

int FirmThresholdCostActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VWidth(ioFlag);
   return status;
}

void FirmThresholdCostActivityBuffer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "VThresh", &mVThresh);
}

void FirmThresholdCostActivityBuffer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "VWidth", &mVWidth, mVWidth);
}

void FirmThresholdCostActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A           = mBufferData.data();
   float const *V     = mInternalState->getBufferData();
   int const nbatch   = getLayerLoc()->nbatch;
   int const nx       = getLayerLoc()->nx;
   int const ny       = getLayerLoc()->ny;
   int const nf       = getLayerLoc()->nf;
   PVHalo const *halo = &getLayerLoc()->halo;
   auto modThresh     = mVThresh + mVWidth;
   auto a2            = 0.5f / modThresh;

   int const numNeuronsAcrossBatch = mInternalState->getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      int kExt = kIndexExtendedBatch(k, nbatch, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      auto input = std::fabs(V[k]);
      auto cost  = input <= modThresh ? input * (1.0f - input * a2) : 0.5f * modThresh;
      A[kExt]    = cost;
   }
}

} // namespace PV
