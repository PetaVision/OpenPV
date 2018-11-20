/*
 * ISTAInternalStateBuffer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTAInternalStateBuffer.hpp"
#include <iostream>

namespace PV {

ISTAInternalStateBuffer::ISTAInternalStateBuffer() {}

ISTAInternalStateBuffer::ISTAInternalStateBuffer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

ISTAInternalStateBuffer::~ISTAInternalStateBuffer() { free(mAdaptiveTimeScaleProbeName); }

void ISTAInternalStateBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   HyPerInternalStateBuffer::initialize(name, params, comm);
}

int ISTAInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_adaptiveTimeScaleProbe(ioFlag);
   return status;
}

void ISTAInternalStateBuffer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "timeConstantTau", &mTimeConstantTau, mTimeConstantTau);
}

void ISTAInternalStateBuffer::ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         name,
         "adaptiveTimeScaleProbe",
         &mAdaptiveTimeScaleProbeName,
         nullptr /*default*/,
         true /*warn if absent*/);
}

Response::Status ISTAInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerInternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *hierarchy = message->mHierarchy;
   if (mAdaptiveTimeScaleProbeName) {
      std::string probeNameString = std::string(mAdaptiveTimeScaleProbeName);
      int maxIterations           = 2;
      auto *namedObject =
            hierarchy->lookupByNameRecursive<Observer>(probeNameString, maxIterations);
      FatalIf(
            namedObject == nullptr,
            "%s adaptiveTimeScaleProbe \"%s\" does not point to an adaptive timescale probe.\n",
            getDescription_c(),
            mAdaptiveTimeScaleProbeName);
      mAdaptiveTimeScaleProbe = dynamic_cast<AdaptiveTimeScaleProbe *>(namedObject);
      FatalIf(
            namedObject == nullptr,
            "%s adaptiveTimeScaleProbe \"%s\" is not an AdaptiveTimeScaleProbe.\n",
            getDescription_c(),
            mAdaptiveTimeScaleProbeName);
   }
   mActivity = hierarchy->lookupByType<ANNActivityBuffer>();
   FatalIf(mActivity == nullptr, "%s needs an ANNActivityBuffer.\n", getDescription_c());
   return Response::SUCCESS;
}

Response::Status ISTAInternalStateBuffer::allocateDataStructures() {
   auto status = HyPerInternalStateBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mAdaptiveTimeScaleProbe) {
      pvAssert(getLayerLoc()->nbatch == mAdaptiveTimeScaleProbe->getNumValues());
   }
   mDeltaTimes.resize(getLayerLoc()->nbatch);

#ifdef PV_USE_CUDA
   if (isUsingGPU()) {
      allocateUpdateKernel();
   }
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

Response::Status
ISTAInternalStateBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = HyPerInternalStateBuffer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   mScaledTimeConstantTau = (float)(mTimeConstantTau / message->mDeltaTime);
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void ISTAInternalStateBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size  = getLayerLoc()->nbatch * sizeof(double);
   mCudaDtAdapt = device->createBuffer(size, &getDescription());
}

void ISTAInternalStateBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mAccumulatedGSyn->isUsingGPU()) {
      mAccumulatedGSyn->copyToCuda();
   }

   // Copy over mCudaDtAdapt
   mCudaDtAdapt->copyToDevice(deltaTimes(simTime, deltaTime));

   // Sync all buffers before running
   mCudaDevice->syncDevice();

   runKernel();
}
#endif // PV_USE_CUDA

void ISTAInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
#ifdef PV_USE_CUDA
   pvAssert(!isUsingGPU()); // or should be in updateBufferGPU() method.
   if (mAccumulatedGSyn->isUsingGPU()) {
      mAccumulatedGSyn->copyFromCuda();
   }
#endif // PV_USE_CUDA

   const PVLayerLoc *loc = getLayerLoc();
   float const *A        = mActivity->getBufferData();
   float *V              = mBufferData.data();

   int nx         = loc->nx;
   int ny         = loc->ny;
   int nf         = loc->nf;
   int numNeurons = getBufferSize();
   int nbatch     = loc->nbatch;
   int lt         = loc->halo.lt;
   int rt         = loc->halo.rt;
   int dn         = loc->halo.dn;
   int up         = loc->halo.up;
   float tau      = mScaledTimeConstantTau;
   float VThresh  = mActivity->getVThresh();

   float const *gSyn     = mAccumulatedGSyn->getBufferData();
   double const *dtAdapt = deltaTimes(simTime, deltaTime);

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons * nbatch; kIndex++) {
      int b                  = kIndex / numNeurons;
      int k                  = kIndex % numNeurons;
      float *VBatch          = V + b * numNeurons;
      float const *gSynBatch = gSyn + b * numNeurons;
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex    = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      float sign = 0.0f;
      if (ABatch[kex] != 0.0f) {
         sign = ABatch[kex] / fabsf(ABatch[kex]);
      }
      VBatch[k] += ((float)dtAdapt[b] / tau) * (gSynBatch[k] - (VThresh * sign));
   }
}

double const *ISTAInternalStateBuffer::deltaTimes(double simTime, double deltaTime) {
   if (mAdaptiveTimeScaleProbe) {
      mAdaptiveTimeScaleProbe->getValues(simTime, &mDeltaTimes);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, deltaTime);
   }
   return mDeltaTimes.data();
}

} /* namespace PV */
