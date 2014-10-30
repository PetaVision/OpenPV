/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

#include "../cudakernels/CudaUpdateHyPerLCALayer.hpp"

#endif

#ifdef __cplusplus
extern "C" {
#endif

void HyPerLCALayer_update_state(
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
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    const bool selfInteract,
    const float dt,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

HyPerLCALayer::HyPerLCALayer()
{
   initialize_base();
}

HyPerLCALayer::HyPerLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

HyPerLCALayer::~HyPerLCALayer()
{
}

int HyPerLCALayer::initialize_base()
{
   numChannels = 1; // If a connection connects to this layer on inhibitory channel, HyPerLayer::requireChannel will add necessary channel
   timeConstantTau = 1.0;
   //Locality in conn
   numWindowX = 1;
   numWindowY = 1;
   windowSymX = false;
   windowSymY = false;
   selfInteract = true;
   sparseProbe = NULL;
   return PV_SUCCESS;
}

bool HyPerLCALayer::inWindowExt(int windowId, int neuronIdxExt){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxExt, loc->nx + loc->halo.lt + loc->halo.rt, loc->ny + loc->halo.dn + loc->halo.up, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxExt, loc->nx + loc->halo.lt + loc->halo.rt, loc->ny + loc->halo.dn + loc->halo.up, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   //std::cout << globalExtX << "/" << loc->nxGlobal + 2*loc->nb << " " << globalExtY << "/" << loc->nyGlobal + 2*loc->nb << " " << outWindow << "," << windowId << "\n";
   return (outWindow == windowId);
}

bool HyPerLCALayer::inWindowRes(int windowId, int neuronIdxRes){
   const PVLayerLoc * loc = this->getLayerLoc();
   int globalExtX = kxPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->kx0;
   int globalExtY = kyPos(neuronIdxRes, loc->nx, loc->ny, loc->nf) + loc->ky0;
   int outWindow = calcWindow(globalExtX, globalExtY);
   return (outWindow == windowId);
}

int HyPerLCALayer::calcWindow(int globalExtX, int globalExtY){
   const PVLayerLoc * loc = this->getLayerLoc();
   //Calculate x and y with symmetry on
   if(windowSymX && globalExtX >= floor((loc->nxGlobal + loc->halo.lt + loc->halo.rt)/2)){
      globalExtX = loc->nxGlobal + loc->halo.lt + loc->halo.rt - globalExtX - 1;
   }
   if(windowSymY && globalExtY >= floor((loc->nyGlobal + loc->halo.dn + loc->halo.up)/2)){
      globalExtY = loc->nyGlobal + loc->halo.dn + loc->halo.up - globalExtY - 1;
   }
   //Calculate the window x and y
   int windowX = floor(((float)globalExtX/(loc->nxGlobal+loc->halo.lt+loc->halo.rt)) * numWindowX);
   int windowY = floor(((float)globalExtY/(loc->nyGlobal+loc->halo.dn+loc->halo.up)) * numWindowY);
   //Change x and y into index
   int windowIdx = (windowY * numWindowY) + windowX;
   assert(windowIdx < getNumWindows());
   assert(windowIdx >= 0);
   return windowIdx;
}

int HyPerLCALayer::getNumWindows(){
   int windowsX, windowsY;
   if(windowSymX){
      windowsX = ceil((float)numWindowX / 2);
   }
   else{
      windowsX = numWindowX;
   }
   if(windowSymY){
      windowsY = ceil((float)numWindowY / 2);
   }
   else{
      windowsY = numWindowY;
   }
   return windowsX * windowsY;
}

int HyPerLCALayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   return PV_SUCCESS;
}

int HyPerLCALayer::allocateDataStructures(){
   int status = ANNLayer::allocateDataStructures();
   return status;
}

int HyPerLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_numChannels(ioFlag);
   ioParam_timeConstantTau(ioFlag);
   ioParam_numWindowX(ioFlag);
   ioParam_numWindowY(ioFlag);
   ioParam_windowSymX(ioFlag);
   ioParam_windowSymY(ioFlag);
   ioParam_selfInteract(ioFlag);
   return status;
}

void HyPerLCALayer::ioParam_numChannels(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numChannels", &numChannels, numChannels, true/*warnIfAbsent*/);
   if (numChannels != 1 && numChannels != 2){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" requires 1 or 2 channels, numChannels = %d\n",
               parent->parameters()->groupKeywordFromName(name), name, numChannels);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
}

void HyPerLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true/*warnIfAbsent*/);
}

void HyPerLCALayer::ioParam_numWindowX(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numWindowX", &numWindowX, numWindowX);
   if(numWindowX != 1) {
      parent->ioParamValue(ioFlag, name, "windowSymX", &windowSymX, windowSymX);
   }
}

void HyPerLCALayer::ioParam_numWindowY(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numWindowY", &numWindowY, numWindowY);
   if(numWindowY != 1) {
      parent->ioParamValue(ioFlag, name, "windowSymY", &windowSymY, windowSymY);
   }
}

void HyPerLCALayer::ioParam_windowSymX(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numWindowX"));
}

void HyPerLCALayer::ioParam_windowSymY(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numWindowY"));
}

void HyPerLCALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (parent->columnId() == 0) {
     std::cout << "selfInteract = " << selfInteract << std::endl;
   }   
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
int HyPerLCALayer::allocateUpdateKernel(){
//#ifdef PV_USE_OPENCL
//   //Not done: what's kernel name for HyPerLCALayer for opencl?
//   int status = CL_SUCCESS;
//   const char* kernel_name = "HyPerLayer_recv_post";
//   char kernelPath[PV_PATH_MAX+128];
//   char kernelFlags[PV_PATH_MAX+128];
//
//   CLDevice * device = parent->getCLDevice();
//
//   sprintf(kernelPath, "%s/../src/kernels/%s.cl", parent->getSrcPath(), kernel_name);
//   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/../src/kernels/", parent->getSrcPath());
//
//   // create kernels
//   krRecvPost = device->createKernel(kernelPath, kernel_name, kernelFlags);
//#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice * device = parent->getCudaDevice();
   //Set to temp pointer of the subclass
   PVCuda::CudaUpdateHyPerLCALayer * updateKernel = new PVCuda::CudaUpdateHyPerLCALayer(device);
   //Set arguments
   const PVLayerLoc* loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int num_neurons = nx*ny*nf;
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
   const bool selfInteract = this->selfInteract;
   //This value is being updated every timestep, so we need to update it on the gpu
   const float dt_tau = -1; //dt/timeConstantTau;
   PVCuda::CudaBuffer* d_GSyn = getDeviceGSyn();
   PVCuda::CudaBuffer* d_activity = getDeviceActivity();

   //Set arguments to kernel
   updateKernel->setArgs(
      num_neurons,
      nx, ny, nf, lt, rt, dn, up,
      numChannels, 
      d_V,
      Vth, AMax, AMin, AShift, VWidth, selfInteract, dt_tau,
      d_GSyn, d_activity);

   //Update d_V for V initialization

   //set updateKernel to krUpdate
   krUpdate = updateKernel;
#endif
   return PV_SUCCESS;
}
#endif


#ifdef PV_USE_CUDA
int HyPerLCALayer::setInitialValues() {
   //Initialize HyPerLayer initial values
   HyPerLayer::setInitialValues();
   //Update V to initial conditions if updating from gpu
   if(updateGpu){
      PVCuda::CudaBuffer* d_V = getDeviceV();
      assert(d_V);
      float* h_V = getV();
      d_V->copyToDevice(h_V);

      PVCuda::CudaBuffer* d_activity = getDeviceActivity();
      assert(d_activity);
      float * h_activity = getCLayer()->activity->data;
      d_activity->copyToDevice(h_activity);
   }
}

int HyPerLCALayer::doUpdateStateGpu(double time, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead){
   //Change dt to match what is passed in
   PVCuda::CudaUpdateHyPerLCALayer* updateKernel = dynamic_cast<PVCuda::CudaUpdateHyPerLCALayer*>(krUpdate);
   assert(updateKernel);
   updateKernel->setDtTau(dt/timeConstantTau);
   runUpdateKernel();
}
#endif

int HyPerLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   //update_timer->start();
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      updateStateOpenCL(time, dt);
//      //HyPerLayer::updateState(time, dt);
//   }
//   else {
//#endif
   {
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      //Only update when the probe updates

      if (triggerLayer != NULL && triggerLayer->needUpdate(time, parent->getDeltaTime())){
         for (int i = 0; i<num_neurons; i++){
            V[i]=0.0;
         }
      }

      HyPerLCALayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, numChannels,
            V, VThresh, AMax, AMin, AShift, VWidth, 
            selfInteract, dt/timeConstantTau, gSynHead, A);
      //if (this->writeSparseActivity){
      //   updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      //}
   }

   //update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLCALayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif


