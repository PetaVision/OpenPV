/*
 * HyPerLCAInternalStateBuffer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCAInternalStateBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "HyPerLCAInternalStateBuffer.kpp"

namespace PV {

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer() {}

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerLCAInternalStateBuffer::~HyPerLCAInternalStateBuffer() { free(mAdaptiveTimeScaleProbeName); }

void HyPerLCAInternalStateBuffer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
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
   auto *objectTable = message->mObjectTable;
   if (mAdaptiveTimeScaleProbeName) {
      mAdaptiveTimeScaleProbe =
            objectTable->findObject<AdaptiveTimeScaleProbe>(mAdaptiveTimeScaleProbeName);
      FatalIf(
            mAdaptiveTimeScaleProbe == nullptr,
            "%s adaptiveTimeScaleProbe \"%s\" is not an AdaptiveTimeScaleProbe.\n",
            getDescription_c(),
            mAdaptiveTimeScaleProbeName);
   }
   mActivity = objectTable->findObject<ActivityBuffer>(getName());
   FatalIf(mActivity == nullptr, "%s could not find an ActivityBuffer.\n", getDescription_c());
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

   PVLayerLoc const *loc        = getLayerLoc();
   int const numNeurons         = getBufferSize();
   double const *dtAdapt        = deltaTimes(simTime, deltaTime);
   float const *accumulatedGSyn = mAccumulatedGSyn->getBufferData();
   float const *A               = mActivity->getBufferData();
   float *V                     = mBufferData.data();

   updateHyPerLCAOnCPU(
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
         dtAdapt,
         mScaledTimeConstantTau,
         accumulatedGSyn,
         A,
         V);
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
