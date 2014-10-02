/*
 * ANNDivInh.cpp
 *
 *  Created on: Jan 22, 2012
 *      Author: kpeterson
 */

#include "ANNDivInh.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNDivLayer_update_state(
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
    float * GSynHead,
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNDivInh::ANNDivInh()
{
   initialize_base();
}

ANNDivInh::ANNDivInh(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNDivInh::~ANNDivInh()
{
}

int ANNDivInh::initialize_base() {
   return PV_SUCCESS;
}

int ANNDivInh::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   numEvents=NUM_ANNDV_EVENTS;
//#endif
   return status;
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int ANNDivInh::initializeThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::initializeThreadBuffers(kernel_name);
//
//   //right now there are no ANN layer specific buffers...
//   return status;
//}
//
//int ANNDivInh::initializeThreadKernels(const char * kernel_name)
//{
////   char kernelPath[256];
////   char kernelFlags[256];
////
////   int status = CL_SUCCESS;
////   CLDevice * device = parent->getCLDevice();
////
////   const char * pvRelPath = "../PetaVision";
////   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getPath(), pvRelPath, kernel_name);
////   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getPath(), pvRelPath);
////
////   // create kernels
////   //
////   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//////kernel name should already be set correctly!
//////   if (spikingFlag) {
//////      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//////   }
//////   else {
//////      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
//////   }
////
////   int argid = 0;
////
////   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
////   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
////   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
////   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
////   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);
////
////
////   status |= krUpdate->setKernelArg(argid++, clV);
////   status |= krUpdate->setKernelArg(argid++, VThresh);
////   status |= krUpdate->setKernelArg(argid++, AMax);
////   status |= krUpdate->setKernelArg(argid++, AMin);
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
//////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
//////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
//////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INHB));
////   status |= krUpdate->setKernelArg(argid++, clActivity);
////
////   return status;
//   return ANNLayer::initializeThreadKernels(kernel_name);
//}
//int ANNDivInh::updateStateOpenCL(double time, double dt)
//{
//   //at the moment there's no reason to do anything differently
//   //for ANNSquaredLayer, but I still defined the method in case
//   //that changes in the future.
//   int status = ANNLayer::updateStateOpenCL(time, dt);
////   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
////   numWait++;
//   return status;
//}
////int ANNDivInh::triggerReceive(InterColComm* comm)
////{
////   int status = HyPerLayer::triggerReceive(comm);
////
////   // copy data to device
////   //
//////   if(gpuAccelerateFlag) {
//////      status |= getChannelCLBuffer()->copyToDevice(&evList[getEVGSyn()]);
//////      numWait += 1;
//////   }
////
////   return status;
////}
//#endif // PV_USE_OPENCL

//! new ANNLayer update state, to add support for GPU kernel.
//
/*!
 * REMARKS:
 *      - This basically will replace the old version of update state
 *        as defined in HyperLayer
 *      - The kernel does the following:
 *      - V = (GSynExc - GSynInh) * (GSynExc - GSynInh)
 *      - Activity = V
 *      - GSynExc = GSynInh = 0
 *
 *
 */
int ANNDivInh::updateState(double time, double dt)
{
   //update_timer->start();
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      updateStateOpenCL(time, dt);
//      //HyPerLayer::updateState(time, dt);
//   }
//   else {
//#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const PVHalo * halo = &(clayer->loc.halo);

      pvdata_t * GSynHead   = GSyn[0];
//      pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//      pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
//      pvdata_t * GSynInhB   = getChannel(CHANNEL_INHB);
      pvdata_t * V = getV();
      pvdata_t * activity = clayer->activity->data;

      ANNDivLayer_update_state(getNumNeurons(), nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, V, VThresh, AMax, AMin, GSynHead, activity);
//#ifdef PV_USE_OPENCL
//   }
//#endif

   //update_timer->stop();
   return PV_SUCCESS;
}

//int ANNDivInh::updateV() {
////   ANNLayer::updateV();
////   squareV();
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
//   pvdata_t * GSynDivInh = this->getChannel(CHANNEL_INHB);
//
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      //V[k] = (GSynExc[k] - GSynInh[k])*(GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
////      printf("V[k] %f\n", V[k]);
////      printf("GSynExc[k] %f\n", GSynExc[k]);
////      printf("GSynInh[k] %f\n", GSynInh[k]);
////      printf("GSynDivInh[k] %f\n", GSynDivInh[k]);
//      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
////      printf("after: V[k] %f\n", V[k]);
//   }
//
//   return PV_SUCCESS;
//}

} /* namespace PV */

///////////////////////////////////////////////////////
//
// implementation of ANNLayer kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNDivLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNDivLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
