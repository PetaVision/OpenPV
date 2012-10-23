/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNSquaredLayer_update_state(
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
//    float * GSynExc,
//    float * GSynInh,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNSquaredLayer::ANNSquaredLayer() {
   initialize_base();
}

// This constructor allows derived classes to set an arbitrary number of channels
ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNSquaredLayer::~ANNSquaredLayer()
{
   // TODO Auto-generated destructor stub
}

int ANNSquaredLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNSquaredLayer::initialize(const char * name, HyPerCol * hc, int numChannels/*Default=MAX_CHANNELS*/) {
   int status = ANNLayer::initialize(name, hc, numChannels);
#ifdef PV_USE_OPENCL
   numEvents=NUM_ANNSQ_EVENTS;
#endif
   return status;
}


#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int ANNSquaredLayer::initializeThreadBuffers(const char * kernel_name)
{
   int status = HyPerLayer::initializeThreadBuffers(kernel_name);

   //right now there are no ANN layer specific buffers...
   return status;
}

int ANNSquaredLayer::initializeThreadKernels(const char * kernel_name)
{
   //at the moment there's no reason to do anything differently
   //for ANNSquaredLayer, but I still defined the method in case
   //that changes in the future.
   return ANNLayer::initializeThreadKernels(kernel_name);
}
int ANNSquaredLayer::updateStateOpenCL(double time, double dt)
{
   //at the moment there's no reason to do anything differently
   //for ANNSquaredLayer, but I still defined the method in case
   //that changes in the future.
   return ANNLayer::updateStateOpenCL(time, dt);
}
#endif

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
int ANNSquaredLayer::updateState(double time, double dt)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(true)) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nb = clayer->loc.nb;

      pvdata_t * GSynHead   = GSyn[0];
//      pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//      pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
      pvdata_t * V = getV();
      pvdata_t * activity = clayer->activity->data;

      ANNSquaredLayer_update_state(getNumNeurons(), nx, ny, nf, nb, V, VThresh, VMax, VMin, GSynHead, activity);
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

//int ANNSquaredLayer::updateV() {
//   ANNLayer::updateV();
//   squareV();
////   pvdata_t * V = getV();
////   pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
////   pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
////   pvdata_t * GSynDivInh = this->getChannel(CHANNEL_INHB);
////
////   for( int k=0; k<getNumNeurons(); k++ ) {
////      //V[k] = (GSynExc[k] - GSynInh[k])*(GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
////      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
////   }
//
//   return PV_SUCCESS;
//}

//int ANNSquaredLayer::squareV() {
//   pvdata_t * V = getV();
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] *= V[k];
//   }
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
#  include "../kernels/ANNSquaredLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNSquaredLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

