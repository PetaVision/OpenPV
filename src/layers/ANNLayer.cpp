/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    float * GSynHead,
/*    float * GSynExc,
    float * GSynInh,*/
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNLayer::ANNLayer() {
   initialize_base();
}

ANNLayer::ANNLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   int status = HyPerLayer::initialize(name, hc, numChannels);
   assert(status == PV_SUCCESS);
   PVParams * params = parent->parameters();

   status |= readVThreshParams(params);
#ifdef PV_USE_OPENCL
   numEvents=NUM_ANN_EVENTS;
#endif
   return status;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int ANNLayer::initializeThreadBuffers(const char * kernel_name)
{
   int status = HyPerLayer::initializeThreadBuffers(kernel_name);

   //right now there are no ANN layer specific buffers...
   return status;
}

int ANNLayer::initializeThreadKernels(const char * kernel_name)
{
   char kernelPath[256];
   char kernelFlags[256];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   const char * pvRelPath = "../PetaVision";
   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getPath(), pvRelPath, kernel_name);
   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getPath(), pvRelPath);

   // create kernels
   //
   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//kernel name should already be set correctly!
//   if (spikingFlag) {
//      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//   }
//   else {
//      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
//   }

   int argid = 0;

   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);


   status |= krUpdate->setKernelArg(argid++, clV);
   status |= krUpdate->setKernelArg(argid++, VThresh);
   status |= krUpdate->setKernelArg(argid++, VMax);
   status |= krUpdate->setKernelArg(argid++, VMin);
   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
   status |= krUpdate->setKernelArg(argid++, clActivity);

   return status;
}
int ANNLayer::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

   // wait for memory to be copied to device
   if (numWait > 0) {
       status |= clWaitForEvents(numWait, evList);
   }
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
   krUpdate->finish();

   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
//   status |= getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(1, &evUpdate, &evList[getEVGSynE()]);
//   status |= getChannelCLBuffer(CHANNEL_INH)->copyFromDevice(1, &evUpdate, &evList[getEVGSynI()]);
   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
   numWait += 2; //3;


   return status;
}
#endif

int ANNLayer::readVThreshParams(PVParams * params) {
   VMax = params->value(name, "VMax", max_pvdata_t);
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   VMin = params->value(name, "VMin", VThresh);
   return PV_SUCCESS;
}

//! new ANNLayer update state, to add support for GPU kernel.
//
/*!
 * REMARKS:
 *      - This basically will replace the old version of update state
 *        as defined in HyperLayer
 *      - The kernel does the following:
 *      - V = GSynExc - GSynInh
 *      - Activity = V
 *      - GSynExc = GSynInh = 0
 *
 *
 */
int ANNLayer::updateState(float time, float dt)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nb = clayer->loc.nb;
      const int numNeurons = getNumNeurons();

      //pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
      //pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
      pvdata_t * GSynHead   = GSyn[0];
      pvdata_t * V = getV();
      pvdata_t * activity = clayer->activity->data;

      ANNLayer_update_state(numNeurons, nx, ny, nf, nb, V, VThresh, VMax, VMin, GSynHead, activity);
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}


//int ANNLayer::updateV() {
//   HyPerLayer::updateV();
//   applyVMax();
//   applyVThresh();
//   return PV_SUCCESS;
//}

//int ANNLayer::applyVMax() {
//   if( VMax < FLT_MAX ) {
//      pvdata_t * V = getV();
//      for( int k=0; k<getNumNeurons(); k++ ) {
//         if(V[k] > VMax) V[k] = VMax;
//      }
//   }
//   return PV_SUCCESS;
//}

//int ANNLayer::applyVThresh() {
//   if( VThresh > -FLT_MIN ) {
//      pvdata_t * V = getV();
//      for( int k=0; k<getNumNeurons(); k++ ) {
//         if(V[k] < VThresh)
//            V[k] = VMin;
//      }
//   }
//   return PV_SUCCESS;
//}


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

