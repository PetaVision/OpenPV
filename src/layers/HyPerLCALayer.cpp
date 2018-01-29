/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

#include "cudakernels/CudaUpdateStateFunctions.hpp"

#endif

void HyPerLCALayer_update_state(
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
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity);

namespace PV {

HyPerLCALayer::HyPerLCALayer() { initialize_base(); }

HyPerLCALayer::HyPerLCALayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

HyPerLCALayer::~HyPerLCALayer() { free(mAdaptiveTimeScaleProbeName); }

int HyPerLCALayer::initialize_base() {
   numChannels = 1; // If a connection connects to this layer on inhibitory channel,
   // HyPerLayer::requireChannel will add necessary channel
   timeConstantTau = 1.0;
   selfInteract    = true;
   return PV_SUCCESS;
}

int HyPerLCALayer::initialize(const char *name, HyPerCol *hc) {
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

Response::Status HyPerLCALayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures(); // Calls allocateUpdateKernel()
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(
         mAdaptiveTimeScaleProbe == nullptr
         || getLayerLoc()->nbatch == mAdaptiveTimeScaleProbe->getNumValues());
   mDeltaTimes.resize(getLayerLoc()->nbatch);
   return Response::SUCCESS;
}

int HyPerLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_selfInteract(ioFlag);
   ioParam_adaptiveTimeScaleProbe(ioFlag);
   return status;
}

void HyPerLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true /*warnIfAbsent*/);
}

void HyPerLCALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (ioFlag == PARAMS_IO_READ && parent->columnId() == 0) {
      InfoLog() << getDescription() << ": selfInteract flag is "
                << (selfInteract ? "true" : "false") << std::endl;
   }
}

void HyPerLCALayer::ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag,
         name,
         "adaptiveTimeScaleProbe",
         &mAdaptiveTimeScaleProbeName,
         nullptr /*default*/,
         true /*warn if absent*/);
}

int HyPerLCALayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   int status = HyPerLayer::requireChannel(channelNeeded, numChannelsResult);
   if (channelNeeded >= 2 && parent->columnId() == 0) {
      WarnLog().printf(
            "HyPerLCALayer \"%s\": connection on channel %d, but HyPerLCA only uses channels 0 and "
            "1.\n",
            name,
            channelNeeded);
   }
   return status;
}

Response::Status
HyPerLCALayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mAdaptiveTimeScaleProbeName) {
      mAdaptiveTimeScaleProbe =
            message->lookup<AdaptiveTimeScaleProbe>(std::string(mAdaptiveTimeScaleProbeName));
      if (mAdaptiveTimeScaleProbe == nullptr) {
         if (parent->getCommunicator()->commRank() == 0) {
            auto isBadType = message->lookup<BaseObject>(std::string(mAdaptiveTimeScaleProbeName));
            if (isBadType != nullptr) {
               ErrorLog() << description << ": adaptiveTimeScaleProbe parameter \""
                          << mAdaptiveTimeScaleProbeName
                          << "\" must be an AdaptiveTimeScaleProbe.\n";
            }
            else {
               ErrorLog() << description << ": adaptiveTimeScaleProbe parameter \""
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

#ifdef PV_USE_CUDA
int HyPerLCALayer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = parent->getDevice();
   // Set to temp pointer of the subclass
   PVCuda::CudaUpdateHyPerLCALayer *updateKernel = new PVCuda::CudaUpdateHyPerLCALayer(device);
   // Set arguments
   const PVLayerLoc *loc   = getLayerLoc();
   const int nx            = loc->nx;
   const int ny            = loc->ny;
   const int nf            = loc->nf;
   const int num_neurons   = nx * ny * nf;
   const int nbatch        = loc->nbatch;
   const int lt            = loc->halo.lt;
   const int rt            = loc->halo.rt;
   const int dn            = loc->halo.dn;
   const int up            = loc->halo.up;
   const int numChannels   = this->numChannels;
   PVCuda::CudaBuffer *d_V = getDeviceV();
   assert(d_V);
   const float Vth         = this->VThresh;
   const float AMax        = this->AMax;
   const float AMin        = this->AMin;
   const float AShift      = this->AShift;
   const float VWidth      = this->VWidth;
   const bool selfInteract = this->selfInteract;
   const float tau         = timeConstantTau
                     / (float)parent->getDeltaTime(); // TODO: eliminate need to call parent method
   PVCuda::CudaBuffer *d_GSyn     = getDeviceGSyn();
   PVCuda::CudaBuffer *d_activity = getDeviceActivity();

   size_t size = parent->getNBatch() * sizeof(double);
   d_dtAdapt   = device->createBuffer(size, &description);

   size        = (size_t)numVertices * sizeof(*verticesV);
   d_verticesV = device->createBuffer(size, &description);
   d_verticesA = device->createBuffer(size, &description);
   d_slopes    = device->createBuffer(size + sizeof(*slopes), &description);

   d_verticesV->copyToDevice(verticesV);
   d_verticesA->copyToDevice(verticesA);
   d_slopes->copyToDevice(slopes);

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
         d_V,
         numVertices,
         d_verticesV,
         d_verticesA,
         d_slopes,
         selfInteract,
         d_dtAdapt,
         tau,
         d_GSyn,
         d_activity);

   // Update d_V for V initialization

   // set updateKernel to krUpdate
   krUpdate = updateKernel;
   return PV_SUCCESS;
}
#endif

#ifdef PV_USE_CUDA
Response::Status HyPerLCALayer::updateStateGpu(double time, double dt) {
   // Copy over d_dtAdapt
   d_dtAdapt->copyToDevice(deltaTimes());
   // Change dt to match what is passed in
   PVCuda::CudaUpdateHyPerLCALayer *updateKernel =
         dynamic_cast<PVCuda::CudaUpdateHyPerLCALayer *>(krUpdate);
   assert(updateKernel);
   runUpdateKernel();
   return Response::SUCCESS;
}
#endif

double HyPerLCALayer::getDeltaUpdateTime() { return parent->getDeltaTime(); }

Response::Status HyPerLCALayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   {
      int nx          = loc->nx;
      int ny          = loc->ny;
      int nf          = loc->nf;
      int num_neurons = nx * ny * nf;
      int nbatch      = loc->nbatch;
      // Only update when the probe updates

      HyPerLCALayer_update_state(
            nbatch,
            num_neurons,
            nx,
            ny,
            nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            numChannels,
            V,
            numVertices,
            verticesV,
            verticesA,
            slopes,
            selfInteract,
            deltaTimes(),
            timeConstantTau / (float)dt,
            gSynHead,
            A);
   }

   return Response::SUCCESS;
}

double *HyPerLCALayer::deltaTimes() {
   if (mAdaptiveTimeScaleProbe) {
      mAdaptiveTimeScaleProbe->getValues(parent->simulationTime(), &mDeltaTimes);
   }
   else {
      mDeltaTimes.assign(getLayerLoc()->nbatch, parent->getDeltaTime());
   }
   return mDeltaTimes.data();
}

} /* namespace PV */

void HyPerLCALayer_update_state(
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
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity) {
   updateV_HyPerLCALayer(
         nbatch,
         numNeurons,
         numChannels,
         V,
         GSynHead,
         activity,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         dtAdapt,
         tau,
         selfInteract,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up);
}
