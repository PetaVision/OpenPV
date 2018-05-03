/*
 * ISTALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

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
   numChannels = 1; // If a connection connects to this layer on inhibitory channel,
   // HyPerLayer::requireChannel will add necessary channel
   timeConstantTau = 1.0f;
   // Locality in conn
   selfInteract = true;
   return PV_SUCCESS;
}

int ISTALayer::initialize(const char *name, HyPerCol *hc) {
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

Response::Status ISTALayer::allocateDataStructures() { return ANNLayer::allocateDataStructures(); }

int ISTALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);

   ioParam_selfInteract(ioFlag);
   return status;
}

void ISTALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true /*warnIfAbsent*/);
}

void ISTALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (parent->columnId() == 0) {
      InfoLog() << "selfInteract = " << selfInteract << std::endl;
   }
}

void ISTALayer::ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag,
         name,
         "adaptiveTimeScaleProbe",
         &mAdaptiveTimeScaleProbeName,
         nullptr /*default*/,
         true /*warn if absent*/);
}

int ISTALayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   int status = HyPerLayer::requireChannel(channelNeeded, numChannelsResult);
   if (channelNeeded >= 2 && parent->columnId() == 0) {
      WarnLog().printf(
            "ISTALayer \"%s\": connection on channel %d, but ISTA only uses channels 0 and 1.\n",
            name,
            channelNeeded);
   }
   return status;
}

#ifdef PV_USE_CUDA
int ISTALayer::allocateUpdateKernel() {
   PVCuda::CudaDevice *device = parent->getDevice();
   // Set to temp pointer of the subclass
   PVCuda::CudaUpdateISTALayer *updateKernel = new PVCuda::CudaUpdateISTALayer(device);
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
         Vth,
         d_dtAdapt,
         tau,
         d_GSyn,
         d_activity);

   krUpdate = updateKernel;
   return PV_SUCCESS;
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

double ISTALayer::getDeltaUpdateTime() { return parent->getDeltaTime(); }

Response::Status ISTALayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   // Only update when the probe updates

   if (triggerLayer != NULL && triggerLayer->needUpdate(time, parent->getDeltaTime())) {
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
         numChannels,
         V,
         VThresh,
         deltaTimes(),
         timeConstantTau / (float)dt,
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
