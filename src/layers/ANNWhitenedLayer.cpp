/*
 * ANNWhitenedLayer.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: garkenyon
 */

#include "ANNWhitenedLayer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ANNWhitenedLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNWhitenedLayer::ANNWhitenedLayer()
{
   initialize_base();
}

ANNWhitenedLayer::ANNWhitenedLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

ANNWhitenedLayer::~ANNWhitenedLayer()
{
}

int ANNWhitenedLayer::initialize_base()
{
   numChannels = 3; // applyGSyn_ANNWhitenedLayer uses 3 channels
   return PV_SUCCESS;
}

int ANNWhitenedLayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);
   assert(numChannels==3);
   return PV_SUCCESS;
}

int ANNWhitenedLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      ANNWhitenedLayer_update_state(num_neurons, nx, ny, nf, loc->nb, V, VThresh, AMax, AMin, AShift, VWidth, gSynHead, A);
      if (this->writeSparseActivity){
         updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      }
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNWhitenedLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNWhitenedLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

