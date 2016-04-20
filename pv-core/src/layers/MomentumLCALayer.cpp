/*
 * MomentumLCALayer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCALayer.hpp"
#include <iostream>

#ifdef PV_USE_CUDA

#include "../cudakernels/CudaUpdateStateFunctions.hpp"

#endif

#ifdef __cplusplus
extern "C" {
#endif

void MomentumLCALayer_update_state(
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
    const float LCAMomentumRate,
    float * GSynHead,
    float * activity,
    float * prevDrive);

#ifdef __cplusplus
}
#endif

namespace PV {

MomentumLCALayer::MomentumLCALayer()
{
   initialize_base();
}

MomentumLCALayer::MomentumLCALayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MomentumLCALayer::~MomentumLCALayer()
{
}

int MomentumLCALayer::initialize_base()
{
   numChannels = 1; // If a connection connects to this layer on inhibitory channel, HyPerLayer::requireChannel will add necessary channel
   timeConstantTau = 1.0;
   LCAMomentumRate = 0;
   //Locality in conn
   //numWindowX = 1;
   //numWindowY = 1;
   //windowSymX = false;
   //windowSymY = false;
   selfInteract = true;
   //sparseProbe = NULL;
   return PV_SUCCESS;
}

int MomentumLCALayer::initialize(const char * name, HyPerCol * hc)
{
   HyPerLCALayer::initialize(name, hc);
   return PV_SUCCESS;
}

int MomentumLCALayer::allocateDataStructures(){
   int status = HyPerLCALayer::allocateDataStructures();
   allocateRestrictedBuffer(&prevDrive, "prevDrive of LCA layer");

   //Initialize buffer to 0
   for(int i = 0; i < getNumNeuronsAllBatches(); i++){
      prevDrive[i] = 0;
   }

#ifdef PV_USE_CUDA
   if(updateGpu){
      d_prevDrive->copyToDevice(prevDrive);
   }
#endif

   return status;
}

int MomentumLCALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLCALayer::ioParamsFillGroup(ioFlag);
   ioParam_LCAMomentumRate(ioFlag);

   return status;
}

void MomentumLCALayer::ioParam_LCAMomentumRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "LCAMomentumRate", &LCAMomentumRate, LCAMomentumRate, true/*warnIfAbsent*/);
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
int MomentumLCALayer::allocateUpdateKernel(){
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice * device = parent->getDevice();
   d_prevDrive = device->createBuffer(getNumNeuronsAllBatches() * sizeof(float));
   //Set to temp pointer of the subclass
   PVCuda::CudaUpdateMomentumLCALayer * updateKernel = new PVCuda::CudaUpdateMomentumLCALayer(device);
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
   assert(d_prevDrive);
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
      d_prevDrive,
      numVertices,
      d_verticesV,
      d_verticesA,
      d_slopes,
      selfInteract, 
      d_dtAdapt,
      tau,
      LCAMomentumRate,
      d_GSyn,
      d_activity);

   //Update d_V for V initialization

   //set updateKernel to krUpdate
   krUpdate = updateKernel;
#endif
   return PV_SUCCESS;
}
#endif
//
//
#ifdef PV_USE_CUDA
int MomentumLCALayer::doUpdateStateGpu(double time, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead){
   if(triggerLayer != NULL){
      fprintf(stderr, "HyPerLayer::Trigger reset of V does not work on GPUs\n");
      abort();
   }
   //Copy over d_dtAdapt
   d_dtAdapt->copyToDevice(parent->getTimeScale());

   //Don't need to copy as prevDrive buffer is only needed for checkpointing
   //d_prevDrive->copyToDevice(prevDrive);

   //Change dt to match what is passed in
   PVCuda::CudaUpdateMomentumLCALayer* updateKernel = dynamic_cast<PVCuda::CudaUpdateMomentumLCALayer*>(krUpdate);
   assert(updateKernel);
   runUpdateKernel();

   //d_prevDrive->copyFromDevice(prevDrive);

   return PV_SUCCESS;
}
#endif

int MomentumLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
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

      MomentumLCALayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, numChannels,
            V, numVertices, verticesV, verticesA, slopes,
            selfInteract, deltaTimeAdapt, timeConstantTau, LCAMomentumRate, gSynHead, A, prevDrive);
      //if (this->writeSparseActivity){
      //   updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      //}
   }

   //update_timer->stop();
   return PV_SUCCESS;
}

int MomentumLCALayer::checkpointWrite(const char * cpDir) {
   HyPerLCALayer::checkpointWrite(cpDir);

#ifdef PV_USE_CUDA
   if(updateGpu){
      d_prevDrive->copyFromDevice(prevDrive);
      parent->getDevice()->syncDevice();
   }
#endif

   // Writes checkpoint files for V, A, and datastore to files in working directory
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   char * filename = NULL;
   filename = parent->pathInCheckpoint(cpDir, getName(), "_prevDrive.pvp");
   int status = writeBufferFile(filename, icComm, timed, &prevDrive, /*numbands*/1, /*extended*/false, getLayerLoc());
   assert(status == PV_SUCCESS);
   free(filename);
   return status;
} /* namespace PV */

int MomentumLCALayer::checkpointRead(const char * cpDir, double * timeptr) {
   HyPerLCALayer::checkpointRead(cpDir, timeptr);
   int status = PV_SUCCESS;
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_prevDrive.pvp");
   status = readBufferFile(filename, parent->icCommunicator(), timeptr, &prevDrive, 1, /*extended*/false, getLayerLoc());
   assert(status == PV_SUCCESS);
   free(filename);



#ifdef PV_USE_CUDA
   //Copy over d_prevDrive
   if(updateGpu){
      d_prevDrive->copyToDevice(prevDrive);
      parent->getDevice()->syncDevice();
   }
#endif
   return status;
}

BaseObject * createMomentumLCALayer(char const * name, HyPerCol * hc) {
   return hc ? new MomentumLCALayer(name, hc) : NULL;
}

}

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/MomentumLCALayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/MomentumLCALayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif



