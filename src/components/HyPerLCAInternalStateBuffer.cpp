/*
 * HyPerLCAInternalStateBuffer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCAInternalStateBuffer.hpp"
#include <iostream>

namespace PV {

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer() {}

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

HyPerLCAInternalStateBuffer::~HyPerLCAInternalStateBuffer() { free(mAdaptiveTimeScaleProbeName); }

void HyPerLCAInternalStateBuffer::initialize(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   HyPerInternalStateBuffer::initialize(name, params, comm);
}

int HyPerLCAInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_selfInteract(ioFlag);
   ioParam_adaptiveTimeScaleProbe(ioFlag);
   return status;
}

void HyPerLCAInternalStateBuffer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "timeConstantTau", &mTimeConstantTau, mTimeConstantTau);
}

void HyPerLCAInternalStateBuffer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "selfInteract", &mSelfInteract, mSelfInteract);
   if (ioFlag == PARAMS_IO_READ && mCommunicator->globalCommRank() == 0) {
      InfoLog() << getDescription() << ": selfInteract flag is "
                << (mSelfInteract ? "true" : "false") << std::endl;
   }
}

void HyPerLCAInternalStateBuffer::ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         name,
         "adaptiveTimeScaleProbe",
         &mAdaptiveTimeScaleProbeName,
         nullptr /*default*/,
         true /*warn if absent*/);
}

Response::Status HyPerLCAInternalStateBuffer::communicateInitInfo(
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
   mActivity = hierarchy->lookupByType<ActivityBuffer>();
   return Response::SUCCESS;
}

Response::Status HyPerLCAInternalStateBuffer::allocateDataStructures() {
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

Response::Status HyPerLCAInternalStateBuffer::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = HyPerInternalStateBuffer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   mScaledTimeConstantTau = (float)(mTimeConstantTau / message->mDeltaTime);
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void HyPerLCAInternalStateBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size  = getLayerLoc()->nbatch * sizeof(double);
   mCudaDtAdapt = device->createBuffer(size, &getDescription());
}

void HyPerLCAInternalStateBuffer::updateBufferGPU(double simTime, double deltaTime) {
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

void HyPerLCAInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
#ifdef PV_USE_CUDA
   pvAssert(!isUsingGPU()); // if using GPU, should be in updateBufferGPU() method instead.
   if (mAccumulatedGSyn->isUsingGPU()) {
      mAccumulatedGSyn->copyFromCuda();
   }
#endif // PV_USE_CUDA

   const PVLayerLoc *loc = getLayerLoc();
   float const *A        = mActivity->getBufferData();
   float *V              = mBufferData.data();

   int nx            = loc->nx;
   int ny            = loc->ny;
   int nf            = loc->nf;
   int numNeurons    = getBufferSize();
   int nbatch        = loc->nbatch;
   int lt            = loc->halo.lt;
   int rt            = loc->halo.rt;
   int dn            = loc->halo.dn;
   int up            = loc->halo.up;
   float tau         = mScaledTimeConstantTau;
   bool selfInteract = mSelfInteract;

   float const *accumulatedGSyn = mAccumulatedGSyn->getBufferData();
   double const *dtAdapt        = deltaTimes(simTime, deltaTime);

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int n = 0; n < numNeurons * nbatch; n++) {
      int b = n / numNeurons; // batch index
      int k = n % numNeurons; // neuron index within batch element

      float exp_tau                     = (float)std::exp(-dtAdapt[b] / (double)tau);
      float *VBatch                     = V + b * numNeurons;
      float const *accumulatedGSynBatch = accumulatedGSyn + b * numNeurons;
      float const gSyn                  = accumulatedGSynBatch[k];
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex   = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
   }
}

double const *HyPerLCAInternalStateBuffer::deltaTimes(double simTime, double deltaTime) {
   if (mAdaptiveTimeScaleProbe) {
      mAdaptiveTimeScaleProbe->getValues(simTime, &mDeltaTimes);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, deltaTime);
   }
   return mDeltaTimes.data();
}

} /* namespace PV */
