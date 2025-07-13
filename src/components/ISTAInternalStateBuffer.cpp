/*
 * ISTAInternalStateBuffer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTAInternalStateBuffer.hpp"

#undef PV_RUN_ON_GPU
#include "ISTAInternalStateBuffer.kpp"

namespace PV {

ISTAInternalStateBuffer::ISTAInternalStateBuffer() {}

ISTAInternalStateBuffer::ISTAInternalStateBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ISTAInternalStateBuffer::~ISTAInternalStateBuffer() {}

void ISTAInternalStateBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerInternalStateBuffer::initialize(params, defaults, comm);
}

int ISTAInternalStateBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerInternalStateBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_timeConstantTau(ioSwitch);
   ioParam_adaptiveTimeScaleProbe(ioSwitch);
   return status;
}

void ISTAInternalStateBuffer::ioParam_timeConstantTau(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "timeConstantTau", &mTimeConstantTau);
}

void ISTAInternalStateBuffer::ioParam_adaptiveTimeScaleProbe(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "adaptiveTimeScaleProbe", &mAdaptiveTimeScaleProbeName);
}

Response::Status ISTAInternalStateBuffer::communicateInitInfo(
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
            "%s could not find an AdaptiveTimeScaleProbe named \"%s\".\n",
            getDescription_c(),
            mAdaptiveTimeScaleProbeName.c_str());
   }
   mActivity = objectTable->findObject<ANNActivityBuffer>(getName());
   FatalIf(mActivity == nullptr, "%s needs an ANNActivityBuffer.\n", getDescription_c());
   return Response::SUCCESS;
}

Response::Status ISTAInternalStateBuffer::allocateDataStructures() {
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
   double const *dtAdapt = deltaTimes(simTime, deltaTime);
   float const *gSyn     = mAccumulatedGSyn->getBufferData();
   float const *A        = mActivity->getBufferData();
   float *V              = mBufferData.data();

   updateISTAInternalStateBufferOnCPU(
         loc->nbatch,
         getBufferSize() /*numNeurons*/,
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         mActivity->getVThresh(),
         dtAdapt,
         mScaledTimeConstantTau,
         gSyn,
         A,
         V);
}

double const *ISTAInternalStateBuffer::deltaTimes(double simTime, double deltaTime) {
   if (mAdaptiveTimeScaleProbe) {
      mDeltaTimes = mAdaptiveTimeScaleProbe->getValues(simTime);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, deltaTime);
   }
   return mDeltaTimes.data();
}

} /* namespace PV */
