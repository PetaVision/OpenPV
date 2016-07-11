/*
 * SpikingLCALayer.cpp
 *
 *  Created on: July 11, 2016
 *      Author: athresher
 */

#include "SpikingLCALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA
#include "../cudakernels/CudaUpdateStateFunctions.hpp"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void SpikingLCALayer_update_state(
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
    float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    const float refactoryScale,
    double * dtAdapt,
    const float tau,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

SpikingLCALayer::SpikingLCALayer()
{
   initialize_base();
}

SpikingLCALayer::SpikingLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

SpikingLCALayer::~SpikingLCALayer() {}

int SpikingLCALayer::initialize_base()
{
   numChannels = 1; // If a connection connects to this layer on inhibitory channel, HyPerLayer::requireChannel will add necessary channel
   timeConstantTau = 1.0;
   refactoryScale = 5.0f;
   return PV_SUCCESS;
}

int SpikingLCALayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

int SpikingLCALayer::allocateDataStructures(){
   int status = ANNLayer::allocateDataStructures();
   return status;
}

int SpikingLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_refactoryScale(ioFlag);
   return status;
}

void SpikingLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true/*warnIfAbsent*/);
}

void SpikingLCALayer::ioParam_refactoryScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "refactoryScale", &refactoryScale, refactoryScale); 
}

int SpikingLCALayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   return HyPerLayer::requireChannel(channelNeeded, numChannelsResult);
}

#if defined(PV_USE_CUDA)
int SpikingLCALayer::allocateUpdateKernel()
{
   PVCuda::CudaDevice * device = parent->getDevice();
   //Set to temp pointer of the subclass
   PVCuda::CudaUpdateSpikingLCALayer * updateKernel = new PVCuda::CudaUpdateSpikingLCALayer(device);
   //Set arguments
   const PVLayerLoc* loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int num_neurons = nx*ny*nf;
   const int nbatch = loc->nbatch;
   const int lt = loc->halo.lt;
   const int rt = loc->halo.rt;
   const int dn = loc->halo.dn;
   const int up = loc->halo.up;
   const int numChannels = this->numChannels;
   PVCuda::CudaBuffer* d_V = getDeviceV();
   assert(d_V);
   const float Vth = this->VThresh;
   const float AMax = this->AMax;
   const float AMin = this->AMin;
   const float AShift = this->AShift;
   const float VWidth = this->VWidth;
   const float refactoryScale = this->refactoryScale;
   //This value is being updated every timestep, so we need to update it on the gpu
   const float tau = timeConstantTau; //dt/timeConstantTau;
   PVCuda::CudaBuffer* d_GSyn = getDeviceGSyn();
   PVCuda::CudaBuffer* d_activity = getDeviceActivity();

   size_t size = parent->getNBatch() * sizeof(double);
   d_dtAdapt = device->createBuffer(size);

   size = (size_t) numVertices * sizeof(*verticesV);
   d_verticesV = device->createBuffer(size);
   d_verticesA = device->createBuffer(size);
   d_slopes = device->createBuffer(size+sizeof(*slopes));

   d_verticesV->copyToDevice(verticesV);
   d_verticesA->copyToDevice(verticesA);
   d_slopes->copyToDevice(slopes);
   

   //Set arguments to kernel
   updateKernel->setArgs(
      nbatch,
      num_neurons,
      nx, ny, nf, lt, rt, dn, up,
      numChannels, 
      d_V,
      numVertices,
      d_verticesV,
      d_verticesA,
      d_slopes,
      refactoryScale, 
      d_dtAdapt,
      tau,
      d_GSyn,
      d_activity);

   //Update d_V for V initialization

   //set updateKernel to krUpdate
   krUpdate = updateKernel;
   return PV_SUCCESS;
}

int SpikingLCALayer::updateStateGpu(double time, double dt)
{
   //Copy over d_dtAdapt
   d_dtAdapt->copyToDevice(parent->getTimeScale());
   //Change dt to match what is passed in
   PVCuda::CudaUpdateSpikingLCALayer* updateKernel = dynamic_cast<PVCuda::CudaUpdateSpikingLCALayer*>(krUpdate);
   assert(updateKernel);
   runUpdateKernel();
   return PV_SUCCESS;
}
#endif

double SpikingLCALayer::getDeltaUpdateTime() { return parent->getDeltaTime(); }

int SpikingLCALayer::updateState(double time, double dt)
{
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = clayer->activity->data;
   pvdata_t * V = getV();
   int num_channels = getNumChannels();
   pvdata_t * gSynHead = GSyn == NULL ? NULL : GSyn[0];

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   
   double * deltaTimeAdapt = parent->getTimeScale();

   SpikingLCALayer_update_state(
         nbatch, num_neurons,
         nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up,
         numChannels, V,
         numVertices, verticesV, verticesA, slopes,
         refactoryScale, deltaTimeAdapt, timeConstantTau,
         gSynHead, A);
   return PV_SUCCESS;
}

BaseObject * createSpikingLCALayer(char const * name, HyPerCol * hc) {
   return hc ? new SpikingLCALayer(name, hc) : NULL;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/SpikingLCALayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/SpikingLCALayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif


