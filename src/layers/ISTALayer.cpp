/*
 * ISTALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

#include "components/TauLayerInputBuffer.hpp"
#include "cudakernels/CudaUpdateStateFunctions.hpp"

#endif

void ISTALayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,

      float *V,
      const float Vth,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity);

namespace PV {

ISTALayer::ISTALayer() { initialize_base(); }

ISTALayer::ISTALayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ISTALayer::~ISTALayer() {}

int ISTALayer::initialize_base() {
   // Locality in conn
   selfInteract = true;
   return PV_SUCCESS;
}

int ISTALayer::initialize(const char *name, HyPerCol *hc) {
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

Response::Status ISTALayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(
         mAdaptiveTimeScaleProbe == nullptr
         || getLayerLoc()->nbatch == mAdaptiveTimeScaleProbe->getNumValues());
   mDeltaTimes.resize(getLayerLoc()->nbatch);

   return Response::SUCCESS;
}

int ISTALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);

   ioParam_selfInteract(ioFlag);
   return status;
}

void ISTALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (parent->getCommunicator()->globalCommRank() == 0) {
      InfoLog() << "selfInteract = " << selfInteract << std::endl;
   }
}

void ISTALayer::ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         name,
         "adaptiveTimeScaleProbe",
         &mAdaptiveTimeScaleProbeName,
         nullptr /*default*/,
         true /*warn if absent*/);
}

LayerInputBuffer *ISTALayer::createLayerInput() { return new TauLayerInputBuffer(name, parent); }

Response::Status
ISTALayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mAdaptiveTimeScaleProbeName) {
      auto *hierarchy             = message->mHierarchy;
      std::string probeNameString = std::string(mAdaptiveTimeScaleProbeName);
      mAdaptiveTimeScaleProbe = hierarchy->lookupByName<AdaptiveTimeScaleProbe>(probeNameString);
      if (mAdaptiveTimeScaleProbe == nullptr) {
         if (parent->getCommunicator()->commRank() == 0) {
            auto isBadType = hierarchy->lookupByName<BaseObject>(probeNameString);
            if (isBadType != nullptr) {
               ErrorLog() << getDescription() << ": adaptiveTimeScaleProbe parameter \""
                          << mAdaptiveTimeScaleProbeName
                          << "\" must be an AdaptiveTimeScaleProbe.\n";
            }
            else {
               ErrorLog() << getDescription() << ": adaptiveTimeScaleProbe parameter \""
                          << mAdaptiveTimeScaleProbeName
                          << "\" is not an AdaptiveTimeScaleProbe in the column.\n";
            }
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   auto status = ANNLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

Response::Status ISTALayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = ANNLayer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *layerInputBuffer = getComponentByType<LayerInputBuffer>();
   pvAssert(layerInputBuffer and layerInputBuffer->getDataStructuresAllocatedFlag());
   double timeConstantTau = layerInputBuffer->getChannelTimeConstant(CHANNEL_EXC);
   scaledTimeConstantTau  = (float)(timeConstantTau / message->mDeltaTime);
   return Response::SUCCESS;
}

void ISTALayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = dt; }

#ifdef PV_USE_CUDA
int ISTALayer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = mCudaDevice;

   size_t size = getLayerLoc()->nbatch * sizeof(double);
   d_dtAdapt   = device->createBuffer(size, &getDescription());

   krUpdate = new PVCuda::CudaUpdateISTALayer(device);

   return PV_SUCCESS;
}

Response::Status ISTALayer::copyInitialStateToGPU() {
   Response::Status status = ANNLayer::copyInitialStateToGPU();
   if (!Response::completed(status)) {
      return status;
   }
   if (!mUpdateGpu) {
      return status;
   }

   // Set arguments of update kernel
   const PVLayerLoc *loc = getLayerLoc();
   const int nx          = loc->nx;
   const int ny          = loc->ny;
   const int nf          = loc->nf;
   const int num_neurons = nx * ny * nf;
   const int nbatch      = loc->nbatch;
   const int lt          = loc->halo.lt;
   const int rt          = loc->halo.rt;
   const int dn          = loc->halo.dn;
   const int up          = loc->halo.up;
   const int numChannels = mLayerInput->getNumChannels();
   pvAssert(mInternalState);
   PVCuda::CudaBuffer *cudaBuffer = mInternalState->getCudaBuffer();
   pvAssert(cudaBuffer);
   const float Vth                          = this->VThresh;
   const float AMax                         = this->AMax;
   const float AMin                         = this->AMin;
   const float AShift                       = this->AShift;
   const float VWidth                       = this->VWidth;
   const bool selfInteract                  = this->selfInteract;
   const float tau                          = scaledTimeConstantTau;
   PVCuda::CudaBuffer *layerInputCudaBuffer = mLayerInput->getCudaBuffer();
   PVCuda::CudaBuffer *activityCudaBuffer   = mActivity->getCudaBuffer();

   auto *updateKernel = dynamic_cast<PVCuda::CudaUpdateISTALayer *>(krUpdate);
   pvAssert(updateKernel);
   // Set arguments to kernel
   updateKernel->setArgs(
         nbatch,
         num_neurons,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         numChannels,
         cudaBuffer,
         Vth,
         d_dtAdapt,
         tau,
         layerInputCudaBuffer,
         activityCudaBuffer);
   return Response::SUCCESS;
}

Response::Status ISTALayer::updateStateGpu(double time, double dt) {
   if (triggerLayer != NULL) {
      Fatal().printf("HyPerLayer::Trigger reset of V does not work on GPUs\n");
   }
   // Copy over d_dtAdapt
   d_dtAdapt->copyToDevice(deltaTimes());
   // Change dt to match what is passed in
   PVCuda::CudaUpdateISTALayer *updateKernel =
         dynamic_cast<PVCuda::CudaUpdateISTALayer *>(krUpdate);
   assert(updateKernel);
   runUpdateKernel();
   return Response::SUCCESS;
}
#endif

Response::Status ISTALayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
   int num_channels      = mLayerInput->getNumChannels();
   float *gSynHead       = mLayerInput->getLayerInput();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   // Only update when the probe updates

   if (triggerLayer != NULL && triggerLayer->needUpdate(time, dt)) {
      for (int i = 0; i < num_neurons * nbatch; i++) {
         V[i] = 0.0;
      }
   }

   ISTALayer_update_state(
         nbatch,
         num_neurons,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         num_channels,
         V,
         VThresh,
         deltaTimes(),
         scaledTimeConstantTau,
         gSynHead,
         A);
   return Response::SUCCESS;
}

double *ISTALayer::deltaTimes() {
   if (mAdaptiveTimeScaleProbe) {
      mAdaptiveTimeScaleProbe->getValues(parent->simulationTime(), &mDeltaTimes);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, parent->getDeltaTime());
   }
   return mDeltaTimes.data();
}

} /* namespace PV */

void ISTALayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,

      float *V,
      const float Vth,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity) {
   updateV_ISTALayer(
         nbatch,
         numNeurons,
         V,
         GSynHead,
         activity,
         Vth,
         dtAdapt,
         tau,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         numChannels);
}
