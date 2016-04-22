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
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
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
      int nbatch = loc->nbatch;
      ANNWhitenedLayer_update_state(nbatch, num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, numVertices, verticesV, verticesA, slopes, gSynHead, A);
//#ifdef PV_USE_OPENCL
//   }
//#endif

   //update_timer->stop();
   return PV_SUCCESS;
}

BaseObject * createANNWhitenedLayer(char const * name, HyPerCol * hc) {
   return hc ? new ANNWhitenedLayer(name, hc) : NULL;
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

