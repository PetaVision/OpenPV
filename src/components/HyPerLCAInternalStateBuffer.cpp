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

HyPerLCAInternalStateBuffer::HyPerLCAInternalStateBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

HyPerLCAInternalStateBuffer::~HyPerLCAInternalStateBuffer() {}

void HyPerLCAInternalStateBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(paramsIO, comm);
}

int HyPerLCAInternalStateBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_timeConstantTau(ioSwitch);
   ioParam_selfInteract(ioSwitch);
   ioParam_adaptiveTimeScaleProbe(ioSwitch);
   return status;
}

void HyPerLCAInternalStateBuffer::ioParam_timeConstantTau(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "timeConstantTau", &mTimeConstantTau);
}

void HyPerLCAInternalStateBuffer::ioParam_selfInteract(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "selfInteract", &mSelfInteract);
   if (ioSwitch == ParamsIOSwitch::Read && mCommunicator->globalCommRank() == 0) {
      InfoLog() << getDescription() << ": selfInteract flag is "
                << (mSelfInteract ? "true" : "false") << std::endl;
   }
}

void HyPerLCAInternalStateBuffer::ioParam_adaptiveTimeScaleProbe(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "adaptiveTimeScaleProbe", &mAdaptiveTimeScaleProbeName);
}

Response::Status HyPerLCAInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerInternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;
   if (!mAdaptiveTimeScaleProbeName.empty()) {
      mAdaptiveTimeScaleProbe =
            objectTable->findObject<AdaptiveTimeScaleProbe>(mAdaptiveTimeScaleProbeName);
      FatalIf(
            mAdaptiveTimeScaleProbe == nullptr,
            "%s adaptiveTimeScaleProbe \"%s\" is not an AdaptiveTimeScaleProbe.\n",
            getDescription_c(),
            mAdaptiveTimeScaleProbeName.c_str());
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
      if (!mAdaptiveTimeScaleProbe->getDataStructuresAllocatedFlag()) {
         InfoLog().printf(
               "%s must wait until %s has finished its allocateDataStructures stage.\n",
               getDescription_c(),
               mAdaptiveTimeScaleProbe->getDescription_c());
         return status + Response::POSTPONE;
      }
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
      mDeltaTimes = mAdaptiveTimeScaleProbe->getValues(simTime);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, deltaTime);
   }
   return mDeltaTimes.data();
}

} /* namespace PV */
