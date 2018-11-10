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

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

HyPerLCAInternalStateBuffer::~HyPerLCAInternalStateBuffer() { free(mAdaptiveTimeScaleProbeName); }

int HyPerLCAInternalStateBuffer::initialize(const char *name, HyPerCol *hc) {
   HyPerInternalStateBuffer::initialize(name, hc);
   return PV_SUCCESS;
}

int HyPerLCAInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_selfInteract(ioFlag);
   ioParam_adaptiveTimeScaleProbe(ioFlag);
   return status;
}

void HyPerLCAInternalStateBuffer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "selfInteract", &mSelfInteract, mSelfInteract);
   if (ioFlag == PARAMS_IO_READ && parent->getCommunicator()->globalCommRank() == 0) {
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
   pvAssert(mLayerInput and mLayerInput->getDataStructuresAllocatedFlag());
   double timeConstantTau = mLayerInput->getChannelTimeConstant(CHANNEL_EXC);
   mScaledTimeConstantTau = (float)(timeConstantTau / message->mDeltaTime);
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void HyPerLCAInternalStateBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size  = getLayerLoc()->nbatch * sizeof(double);
   mCudaDtAdapt = device->createBuffer(size, &getDescription());

   mCudaUpdateKernel = new PVCuda::CudaUpdateHyPerLCAInternalState(device);
}

Response::Status HyPerLCAInternalStateBuffer::copyInitialStateToGPU() {
   Response::Status status = HyPerInternalStateBuffer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!isUsingGPU()) {
      return status;
   }

   // Set arguments of update kernel
   const PVLayerLoc *loc = getLayerLoc();
   int const nx          = loc->nx;
   int const ny          = loc->ny;
   int const nf          = loc->nf;
   int const numNeurons  = nx * ny * nf;
   int const nbatch      = loc->nbatch;
   int const lt          = loc->halo.lt;
   int const rt          = loc->halo.rt;
   int const dn          = loc->halo.dn;
   int const up          = loc->halo.up;
   int const numChannels = mLayerInput->getNumChannels();
   pvAssert(getCudaBuffer());
   float const selfInteract                 = (float)this->mSelfInteract;
   float const tau                          = mScaledTimeConstantTau;
   PVCuda::CudaBuffer *layerInputCudaBuffer = mLayerInput->getCudaBuffer();
   PVCuda::CudaBuffer *activityCudaBuffer   = mActivity->getCudaBuffer();
   pvAssert(layerInputCudaBuffer);

   auto *cudaKernel = dynamic_cast<PVCuda::CudaUpdateHyPerLCAInternalState *>(mCudaUpdateKernel);
   pvAssert(cudaKernel);
   // Set arguments to kernel
   cudaKernel->setArgs(
         nbatch,
         numNeurons,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         numChannels,
         getCudaBuffer(),
         selfInteract,
         mCudaDtAdapt,
         tau,
         layerInputCudaBuffer,
         activityCudaBuffer);
   return Response::SUCCESS;
}

void HyPerLCAInternalStateBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // or should be in updateBufferCPU() method.
   if (!mLayerInput->isUsingGPU()) {
      mLayerInput->copyToCuda();
   }

   // Copy over mCudaDtAdapt
   mCudaDtAdapt->copyToDevice(deltaTimes(simTime, deltaTime));

   // Sync all buffers before running
   mCudaDevice->syncDevice();

   // Run kernel
   mCudaUpdateKernel->run();
}
#endif // PV_USE_CUDA

void HyPerLCAInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
#ifdef PV_USE_CUDA
   pvAssert(!isUsingGPU()); // or should be in updateBufferGPU() method.
   if (mLayerInput->isUsingGPU()) {
      mLayerInput->copyFromCuda();
   }
#endif // PV_USE_CUDA

   const PVLayerLoc *loc = getLayerLoc();
   float const *A        = mActivity->getBufferData();
   float *V              = mBufferData.data();
   int numChannels       = mLayerInput->getNumChannels();

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

   float const *GSynExc  = mLayerInput->getChannelData(CHANNEL_EXC);
   double const *dtAdapt = deltaTimes(simTime, deltaTime);

   if (numChannels == 1) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int kIndex = 0; kIndex < numNeurons * nbatch; kIndex++) {
         int b = kIndex / numNeurons;
         int k = kIndex % numNeurons;

         float exp_tau             = (float)std::exp(-dtAdapt[b] / (double)tau);
         float *VBatch             = V + b * numNeurons;
         float const *GSynExcBatch = GSynExc + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         float const gSyn = GSynExcBatch[k]; // only one channel
         int kex          = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
      }
   }
   else {
      pvAssert(numChannels > 1);
      float const *GSynInh = mLayerInput->getChannelData(CHANNEL_INH);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int kIndex = 0; kIndex < numNeurons * nbatch; kIndex++) {
         int b = kIndex / numNeurons;
         int k = kIndex % numNeurons;

         float exp_tau             = (float)std::exp(-dtAdapt[b] / (double)tau);
         float *VBatch             = V + b * numNeurons;
         float const *GSynExcBatch = GSynExc + b * numNeurons;
         float const *GSynInhBatch = GSynInh + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         float const gSyn = GSynExcBatch[k] - GSynInhBatch[k];
         int kex          = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
      }
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
