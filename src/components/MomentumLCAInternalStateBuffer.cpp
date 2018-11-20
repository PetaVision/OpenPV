/*
 * MomentumLCAInternalStateBuffer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCAInternalStateBuffer.hpp"
#include <iostream>

namespace PV {

MomentumLCAInternalStateBuffer::MomentumLCAInternalStateBuffer() {}

MomentumLCAInternalStateBuffer::MomentumLCAInternalStateBuffer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

MomentumLCAInternalStateBuffer::~MomentumLCAInternalStateBuffer() {}

int MomentumLCAInternalStateBuffer::initialize(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   HyPerLCAInternalStateBuffer::initialize(name, params, comm);
   return PV_SUCCESS;
}

int MomentumLCAInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLCAInternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_LCAMomentumRate(ioFlag);
   return status;
}

void MomentumLCAInternalStateBuffer::ioParam_LCAMomentumRate(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "LCAMomentumRate",
         &mLCAMomentumRate,
         mLCAMomentumRate,
         true /*warnIfAbsent*/);
}

Response::Status MomentumLCAInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLCAInternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *hierarchy           = message->mHierarchy;
   std::string prevDriveName = std::string("prevDrive \"") + getName() + "\"";
   mPrevDrive                = hierarchy->lookupByName<RestrictedBuffer>(prevDriveName);
   FatalIf(
         mPrevDrive == nullptr,
         "%s requires a RestrictedBuffer with the label \"prevDrive\" and the name \"%s\".\n",
         getDescription_c(),
         getName());
   return Response::SUCCESS;
}

Response::Status MomentumLCAInternalStateBuffer::allocateDataStructures() {
   return HyPerLCAInternalStateBuffer::allocateDataStructures();
}

Response::Status MomentumLCAInternalStateBuffer::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   return HyPerLCAInternalStateBuffer::initializeState(message);
}

#ifdef PV_USE_CUDA
void MomentumLCAInternalStateBuffer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size  = getLayerLoc()->nbatch * sizeof(double);
   mCudaDtAdapt = device->createBuffer(size, &getDescription());
}

void MomentumLCAInternalStateBuffer::updateBufferGPU(double simTime, double deltaTime) {
   pvAssert(isUsingGPU()); // if not using GPU, should be in updateBufferCPU() method instead.
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

void MomentumLCAInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
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

   float const *gSyn     = mAccumulatedGSyn->getBufferData();
   double const *dtAdapt = deltaTimes(simTime, deltaTime);
   float *prevDrive      = mPrevDrive->getReadWritePointer();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kIndex = 0; kIndex < numNeurons * nbatch; kIndex++) {
      int b = kIndex / numNeurons;
      int k = kIndex % numNeurons;

      float exp_tau          = (float)std::exp(-dtAdapt[b] / (double)tau);
      float *VBatch          = V + b * numNeurons;
      float const *gSynBatch = gSyn + b * numNeurons;
      float *prevDriveBatch  = prevDrive + b * numNeurons;
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex            = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      float currentDrive = (1.0f - exp_tau) * (gSynBatch[k] + selfInteract * ABatch[kex]);
      // Accumulate into VBatch with decay and momentum
      VBatch[k] = exp_tau * VBatch[k] + currentDrive + mLCAMomentumRate * prevDriveBatch[k];
      // Update momentum buffer
      prevDriveBatch[k] = currentDrive;
   }
}

} /* namespace PV */
