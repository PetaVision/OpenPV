/*
 * MomentumLCAInternalStateBuffer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCAInternalStateBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "MomentumLCAInternalStateBuffer.kpp"

namespace PV {

MomentumLCAInternalStateBuffer::MomentumLCAInternalStateBuffer() {}

MomentumLCAInternalStateBuffer::MomentumLCAInternalStateBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

MomentumLCAInternalStateBuffer::~MomentumLCAInternalStateBuffer() {}

int MomentumLCAInternalStateBuffer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
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
   auto *objectTable = message->mObjectTable;
   char const *label = "prevDrive";
   auto buffers      = objectTable->findObjects<RestrictedBuffer>(getName());
   mPrevDrive        = nullptr;
   for (auto &buf : buffers) {
      if (buf->getBufferLabel() == label) {
         FatalIf(
               mPrevDrive != nullptr,
               "%s found multiple buffers with name \%s\" and label \"%s\"\n",
               getDescription_c(),
               getName(),
               label);
         mPrevDrive = buf;
      }
   }
   FatalIf(
         mPrevDrive == nullptr,
         "%s could not find a RestrictedBuffer with name \"%s\" and label \"%s\".\n",
         getDescription_c(),
         getName(),
         label);
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
   int numNeurons        = getBufferSize();

   double const *dtAdapt        = deltaTimes(simTime, deltaTime);
   float const *accumulatedGSyn = mAccumulatedGSyn->getBufferData();
   float const *A               = mActivity->getBufferData();
   float *prevDrive             = mPrevDrive->getReadWritePointer();
   float *V                     = mBufferData.data();

   updateMomentumLCAOnCPU(
         loc->nbatch,
         numNeurons,
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         mSelfInteract,
         mLCAMomentumRate,
         dtAdapt,
         mScaledTimeConstantTau,
         accumulatedGSyn,
         A,
         prevDrive,
         V);
}

} /* namespace PV */
