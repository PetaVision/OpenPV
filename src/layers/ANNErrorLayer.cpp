/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNErrorLayer_update_state(
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
    float * GSynHead,
    float * activity,
    float errScale);


#ifdef __cplusplus
}
#endif

namespace PV {

ANNErrorLayer::ANNErrorLayer()
{
   initialize_base();
}

ANNErrorLayer::ANNErrorLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

ANNErrorLayer::~ANNErrorLayer()
{
}

int ANNErrorLayer::initialize_base()
{
   errScale = 1;
   return PV_SUCCESS;
}

int ANNErrorLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int ANNErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   return status;
}

void ANNErrorLayer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "errScale", &errScale, errScale, true/*warnIfAbsent*/);
}

int ANNErrorLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   update_timer->start();
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
         ANNErrorLayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, VThresh,
               AMax, AMin, AShift, gSynHead, A, errScale);
//#ifdef PV_USE_OPENCL
//   }
//#endif

   update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNErrorLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNErrorLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
