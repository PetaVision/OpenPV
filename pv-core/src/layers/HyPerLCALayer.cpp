/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

#include "../cudakernels/CudaUpdateStateFunctions.hpp"

#endif

#ifdef __cplusplus
extern "C" {
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
    float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    const bool selfInteract,
    double * dtAdapt,
    const float tau,
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
   //numWindowX = 1;
   //numWindowY = 1;
   //windowSymX = false;
   //windowSymY = false;
   selfInteract = true;
   //sparseProbe = NULL;
   return PV_SUCCESS;
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
   ioParam_numChannels(ioFlag);  // Deprecated Jul 9, 2015.  All ioParam_numChannels does is issue a warning that numChannels is no longer used.  Delete after a suitable fade time.
   ioParam_timeConstantTau(ioFlag);
#ifdef OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.
   ioParam_numWindowX(ioFlag);
   ioParam_numWindowY(ioFlag);
   ioParam_windowSymX(ioFlag);
   ioParam_windowSymY(ioFlag);
#endif // OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.

   ioParam_selfInteract(ioFlag);
   return status;
}

// After a suitable fade time, HyPerLCALayer::ioParam_numChannels() can be removed
void HyPerLCALayer::ioParam_numChannels(enum ParamsIOFlag ioFlag) {
   if (parent->parameters()->present(name, "numChannels")) {
      if ( parent->columnId()==0) {
         fprintf(stderr, "HyPerLCALayer \"%s\" warning: the parameter numChannels is no longer used; connections that connect to the layer create channels as needed.\n", name);
      }
      parent->parameters()->value(name, "numChannels"); // mark the parameter as read
   }
#ifdef OBSOLETE // Marked obsolete Jul 9, 2015.  A layer learns how many channels it has during the communication stage.
   parent->ioParamValue(ioFlag, name, "numChannels", &numChannels, numChannels, true/*warnIfAbsent*/);
   if (numChannels != 1 && numChannels != 2){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" requires 1 or 2 channels, numChannels = %d\n",
               getKeyword(), name, numChannels);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
#endif // OBSOLETE // Marked obsolete Jul 9, 2015.  A layer learns how many channels it has during the communication stage.
}

void HyPerLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true/*warnIfAbsent*/);
}

#ifdef OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.
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
#endif // OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.

void HyPerLCALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (ioFlag==PARAMS_IO_READ && parent->columnId() == 0) {
     std::cout << getKeyword() << "\"" << name << "\"" << ": selfInteract flag is " << (selfInteract ? "true" : "false") << std::endl;
   }   
}

int HyPerLCALayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   int status = HyPerLayer::requireChannel(channelNeeded, numChannelsResult);
   if (channelNeeded>=2 && parent->columnId()==0) {
      fprintf(stderr, "HyPerLCALayer \"%s\" warning: connection on channel %d, but HyPerLCA only uses channels 0 and 1.\n", name, channelNeeded);
   }
   return status;
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
   PVCuda::CudaDevice * device = parent->getDevice();
   //Set to temp pointer of the subclass
   PVCuda::CudaUpdateHyPerLCALayer * updateKernel = new PVCuda::CudaUpdateHyPerLCALayer(device);
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
   const bool selfInteract = this->selfInteract;
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
      selfInteract, 
      d_dtAdapt,
      tau,
      d_GSyn,
      d_activity);

   //Update d_V for V initialization

   //set updateKernel to krUpdate
   krUpdate = updateKernel;
#endif
   return PV_SUCCESS;
}
#endif


#ifdef PV_USE_CUDA
int HyPerLCALayer::doUpdateStateGpu(double time, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead){
   if(triggerLayer != NULL){
      fprintf(stderr, "HyPerLayer::Trigger reset of V does not work on GPUs\n");
      abort();
   }
   //Copy over d_dtAdapt
   d_dtAdapt->copyToDevice(parent->getTimeScale());
   //Change dt to match what is passed in
   PVCuda::CudaUpdateHyPerLCALayer* updateKernel = dynamic_cast<PVCuda::CudaUpdateHyPerLCALayer*>(krUpdate);
   assert(updateKernel);
   runUpdateKernel();
   return PV_SUCCESS;
}
#endif

double HyPerLCALayer::getDeltaUpdateTime(){
   return parent->getDeltaTime();
}

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
      int nbatch = loc->nbatch;
      //Only update when the probe updates
      
      double * deltaTimeAdapt = parent->getTimeScale();

      HyPerLCALayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, numChannels,
            V, numVertices, verticesV, verticesA, slopes,
            selfInteract, deltaTimeAdapt, timeConstantTau, gSynHead, A);
      //if (this->writeSparseActivity){
      //   updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      //}
   }

   //update_timer->stop();
   return PV_SUCCESS;
}

BaseObject * createHyPerLCALayer(char const * name, HyPerCol * hc) {
   return hc ? new HyPerLCALayer(name, hc) : NULL;
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


