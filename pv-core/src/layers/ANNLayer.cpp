/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "../layers/updateStateFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

void ANNLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    int num_channels,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNLayer::ANNLayer() {
   initialize_base();
}

ANNLayer::ANNLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   assert(status == PV_SUCCESS);

   status |= checkVThreshParams(parent->parameters());
//#ifdef PV_USE_OPENCL
//   numEvents=NUM_ANN_EVENTS;
//#endif
   return status;
}

int ANNLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VMin(ioFlag);
   ioParam_VMax(ioFlag);
   ioParam_VShift(ioFlag);
   ioParam_VWidth(ioFlag);
   ioParam_clearGSynInterval(ioFlag);

   if (ioFlag == PARAMS_IO_READ) {
      status = checkVThreshParams(parent->parameters());
   }
   return status;
}

void ANNLayer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VThresh", &VThresh, -max_pvvdata_t);
}

// Parameter VMin was deprecated in favor of AMin on Mar 20, 2014
void ANNLayer::ioParam_VMin(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VMin")) {
      AMin = parent->parameters()->value(name, "VMin");
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: %s \"%s\" parameter \"VMin\" is deprecated.  Use AMin instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      return;
   }
   parent->ioParamValue(ioFlag, name, "AMin", &AMin, VThresh);
}

// Parameter VMax was deprecated in favor of AMax on Mar 20, 2014
void ANNLayer::ioParam_VMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VMax")) {
      AMax = parent->parameters()->value(name, "VMax");
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: %s \"%s\" parameter \"VMax\" is deprecated.  Use AMax instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      return;
   }
   parent->ioParamValue(ioFlag, name, "AMax", &AMax, max_pvvdata_t);
}

void ANNLayer::ioParam_VShift(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "VShift")) {
      AShift = parent->parameters()->value(name, "VShift");
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: %s \"%s\" parameter \"VShift\" is deprecated.  Use AShift instead.\n",
               parent->parameters()->groupKeywordFromName(name), name);
      }
      return;
   }
   parent->ioParamValue(ioFlag, name, "AShift", &AShift, (pvdata_t) 0);
}

void ANNLayer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VWidth", &VWidth, (pvdata_t) 0);
}

void ANNLayer::ioParam_clearGSynInterval(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "clearGSynInterval", &clearGSynInterval, 0.0);
   if (ioFlag==PARAMS_IO_READ) {
      nextGSynClearTime = parent->getStartTime();
   }
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int ANNLayer::allocateThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::allocateThreadBuffers(kernel_name);
//
//   //right now there are no ANN layer specific buffers...
//   return status;
//}
//
//int ANNLayer::initializeThreadKernels(const char * kernel_name)
//{
//   char kernelPath[256];
//   char kernelFlags[256];
//
//   int status = CL_SUCCESS;
//   CLDevice * device = parent->getCLDevice();
//
//   const char * pvRelPath = "../PetaVision";
//   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getSrcPath(), pvRelPath, kernel_name);
//   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getSrcPath(), pvRelPath);
//
//   // create kernels
//   //
//   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////kernel name should already be set correctly!
////   if (spikingFlag) {
////      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////   }
////   else {
////      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
////   }
//
//   int argid = 0;
//
//   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);
//
//
//   status |= krUpdate->setKernelArg(argid++, clV);
//   status |= krUpdate->setKernelArg(argid++, VThresh);
//   status |= krUpdate->setKernelArg(argid++, AMax);
//   status |= krUpdate->setKernelArg(argid++, AMin);
//   status |= krUpdate->setKernelArg(argid++, AShift);
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
//   status |= krUpdate->setKernelArg(argid++, clActivity);
//
//   return status;
//}
//int ANNLayer::updateStateOpenCL(double time, double dt)
//{
//   int status = CL_SUCCESS;
//
//   // wait for memory to be copied to device
//   if (numWait > 0) {
//       status |= clWaitForEvents(numWait, evList);
//   }
//   for (int i = 0; i < numWait; i++) {
//      clReleaseEvent(evList[i]);
//   }
//   numWait = 0;
//
//   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
//   krUpdate->finish();
//
//   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
//   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
//   numWait += 2; //3;
//
//
//   return status;
//}
//#endif

int ANNLayer::checkVThreshParams(PVParams * params) {
   if (VWidth<0) {
      VThresh += VWidth;
      VWidth = -VWidth;
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" warning: interpreting negative VWidth as setting VThresh=%f and VWidth=%f\n",
               parent->parameters()->groupKeywordFromName(name), name, VThresh, VWidth);
      }
   }

   pvdata_t limfromright = VThresh+VWidth-AShift;
   if (AMax < limfromright) limfromright = AMax;

   if (AMin > limfromright) {
      if (parent->columnId()==0) {
         if (VWidth==0) {
            fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, jumping from %f to %f at Vthresh=%f\n",
                  parent->parameters()->groupKeywordFromName(name), name, AMin, limfromright, VThresh);
         }
         else {
            fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, changing from %f to %f as V goes from VThresh=%f to VThresh+VWidth=%f\n",
                  parent->parameters()->groupKeywordFromName(name), name, AMin, limfromright, VThresh, VThresh+VWidth);
         }
      }
   }
   return PV_SUCCESS;
}

int ANNLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   bool clearNow = clearGSynInterval <= 0 || timef >= nextGSynClearTime;
   if (clearNow) {
      resetGSynBuffers_HyPerLayer(this->getNumNeurons(), getNumChannels(), GSyn[0]);
   }
   if (clearNow > 0) {
      nextGSynClearTime += clearGSynInterval;   
   }
   return status;
}

//! new ANNLayer update state, to add support for GPU kernel.
//
/*!
 * REMARKS:
 *      - The kernel does the following:
//   HyPerLayer::updateV();
 *      - V = GSynExc - GSynInh
//   applyVMax(); (see below)
//   applyVThresh(); (see below)
 *      - Activity = V
 *
 *
 */
int ANNLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
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
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      ANNLayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, VThresh, AMax, AMin, AShift, VWidth, num_channels, gSynHead, A);

      //Done in UpdateState
      
      //if (this->writeSparseActivity){
      //   updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      //}
//#ifdef PV_USE_OPENCL
//   }
//#endif

   //update_timer->stop();
   return PV_SUCCESS;
}

int ANNLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   PVHalo const * halo = &loc->halo;
   int num_neurons = nx*ny*nf;
   int status;
   status = setActivity_HyPerLayer(num_neurons, getCLayer()->activity->data, getV(), nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(num_neurons, getV(), AMin, VThresh, AShift, VWidth, getCLayer()->activity->data, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(num_neurons, getV(), AMax, getCLayer()->activity->data, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
   return status;
}

int ANNLayer::checkpointRead(char const * cpDir, double * timeptr) {
   int status = HyPerLayer::checkpointRead(cpDir, timeptr);
   if (status==PV_SUCCESS) {
      status = parent->readScalarFromFile(cpDir, getName(), "nextGSynClearTime", &nextGSynClearTime, parent->simulationTime()-parent->getDeltaTime());
   }
   return status;
}

int ANNLayer::checkpointWrite(char const * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   if (status==PV_SUCCESS) {
      status = parent->writeScalarToFile(cpDir, getName(), "nextGSynClearTime", nextGSynClearTime);
   }
   return status;
}


}  // end namespace PV

///////////////////////////////////////////////////////
//
// implementation of ANNLayer kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

