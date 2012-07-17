/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "BIDSSourceLayer.hpp"

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

BIDSSourceLayer::BIDSSourceLayer() {
   initialize_base();
}

BIDSSourceLayer::BIDSSourceLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

BIDSSourceLayer::~BIDSSourceLayer() {}

int BIDSSourceLayer::initialize_base() {
   return PV_SUCCESS;
}

int BIDSSourceLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   int status = ANNLayer::initialize(name, hc, numChannels);
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
int BIDSSourceLayer::initializeThreadBuffers(const char * kernel_name)
{
   int status = HyPerLayer::initializeThreadBuffers(kernel_name);

   //right now there are no ANN layer specific buffers...
   return status;
}

int BIDSSourceLayer::initializeThreadKernels(const char * kernel_name)
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
int BIDSSourceLayer::updateStateOpenCL(float time, float dt)
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

int BIDSSourceLayer::readVThreshParams(PVParams * params) {
   VMax = params->value(name, "VMax", max_pvdata_t);
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   VMin = params->value(name, "VMin", VThresh);
   return PV_SUCCESS;
}

int BIDSSourceLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
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
int BIDSSourceLayer::updateState(float time, float dt)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      const int numNeurons = getNumNeurons();
      float * V = getV();
      for(int i = 0; i < numNeurons; i++){
         V[i] = VThresh;
      }
      const PVLayerLoc * loc = getLayerLoc();
      setActivity_HyPerLayer(getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);

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
//#  include "../kernels/ANNLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
//#  include "../kernels/ANNLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

