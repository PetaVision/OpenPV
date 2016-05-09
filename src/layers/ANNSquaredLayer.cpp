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
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNSquaredLayer::ANNSquaredLayer() {
   initialize_base();
}

ANNSquaredLayer::ANNSquaredLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}  // end V1AveSquaredInput::V1AveSquaredInput(const char *, HyPerCol *)

ANNSquaredLayer::~ANNSquaredLayer()
{
}

int ANNSquaredLayer::initialize_base() {
   numChannels = 1; // ANNSquaredLayer only takes input on the excitatory channel
   return PV_SUCCESS;
}

int ANNSquaredLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels==1);
//#ifdef PV_USE_OPENCL
//   numEvents=NUM_ANNSQ_EVENTS;
//#endif
   return status;
}

int ANNSquaredLayer::updateState(double time, double dt)
{
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nbatch = clayer->loc.nbatch;

      pvdata_t * GSynHead   = GSyn[0];
      pvdata_t * V = getV();
      pvdata_t * activity = clayer->activity->data;

      ANNSquaredLayer_update_state(nbatch, getNumNeurons(), nx, ny, nf, clayer->loc.halo.lt, clayer->loc.halo.rt, clayer->loc.halo.dn, clayer->loc.halo.up, V, GSynHead, activity);
//#ifdef PV_USE_OPENCL
//   }
//#endif

   //update_timer->stop();
   return PV_SUCCESS;
}

BaseObject * createANNSquaredLayer(char const * name, HyPerCol * hc) {
   return hc ? new ANNSquaredLayer(name, hc) : NULL;
}

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

