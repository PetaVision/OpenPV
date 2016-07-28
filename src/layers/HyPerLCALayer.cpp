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
#ifdef OBSOLETE // Marked obsolete Jun 27, 2016.
   ioParam_numChannels(ioFlag);  // Deprecated Jul 9, 2015.  All ioParam_numChannels does is issue a warning that numChannels is no longer used.  Delete after a suitable fade time.
#endif // OBSOLETE // Marked obsolete Jun 27, 2016.
   ioParam_timeConstantTau(ioFlag);
   ioParam_selfInteract(ioFlag);
   return status;
}

#ifdef OBSOLETE // Marked obsolete Jun 27, 2016.
void HyPerLCALayer::ioParam_numChannels(enum ParamsIOFlag ioFlag) {
   if (parent->parameters()->present(name, "numChannels")) {
      if ( parent->columnId()==0) {
         pvWarn().printf("HyPerLCALayer \"%s\": the parameter numChannels is no longer used; connections that connect to the layer create channels as needed.\n", name);
      }
      parent->parameters()->value(name, "numChannels"); // mark the parameter as read
   }
}
#endif // OBSOLETE // Marked obsolete Jun 27, 2016.

void HyPerLCALayer::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "timeConstantTau", &timeConstantTau, timeConstantTau, true/*warnIfAbsent*/);
}

void HyPerLCALayer::ioParam_selfInteract(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "selfInteract", &selfInteract, selfInteract);
   if (ioFlag==PARAMS_IO_READ && parent->columnId() == 0) {
      pvInfo() << getDescription() << ": selfInteract flag is " << (selfInteract ? "true" : "false") << std::endl;
   }   
}

int HyPerLCALayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   int status = HyPerLayer::requireChannel(channelNeeded, numChannelsResult);
   if (channelNeeded>=2 && parent->columnId()==0) {
      pvWarn().printf("HyPerLCALayer \"%s\": connection on channel %d, but HyPerLCA only uses channels 0 and 1.\n", name, channelNeeded);
   }
   return status;
}


#ifdef PV_USE_CUDA
int HyPerLCALayer::allocateUpdateKernel(){
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
   return PV_SUCCESS;
}
#endif


#ifdef PV_USE_CUDA
int HyPerLCALayer::updateStateGpu(double time, double dt)
{
  //this is a change
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

int HyPerLCALayer::updateState(double time, double dt)
{
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = clayer->activity->data;
   pvdata_t * V = getV();
   int num_channels = getNumChannels();
   pvdata_t * gSynHead = GSyn == NULL ? NULL : GSyn[0];
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
   }

   return PV_SUCCESS;
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

    float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    const bool selfInteract,
    double* dtAdapt,
    const float tau,
    float * GSynHead,
    float * activity)
{
   updateV_HyPerLCALayer(nbatch, numNeurons, numChannels, V, GSynHead, activity,
		   numVertices, verticesV, verticesA, slopes, dtAdapt, tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
}
