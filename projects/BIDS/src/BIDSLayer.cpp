/*
 * BIDSLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 */

#include <layers/HyPerLayer.hpp>
#include "BIDSLayer.hpp"
#include <layers/LIF.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#ifdef __cplusplus
extern "C" {
#endif

void LIF_update_state_arma(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
    float * activity);

void LIF_update_state_beginning(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
    float * activity);

void LIF_update_state_original(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {
BIDSLayer::BIDSLayer() {
  // initialize(arguments) should *not* be called by the protected constructor.
}

BIDSLayer::BIDSLayer(const char * name, HyPerCol * hc) {
   initialize(name, hc, "BIDS_update_state");
}

int BIDSLayer::initialize(const char * name, HyPerCol * hc, const char * kernel_name){
   LIF::initialize(name, hc, "BIDS_update_state");
   assert(numChannels==3);
   return PV_SUCCESS;
}

int BIDSLayer::updateState(double time, double dt)
{
   int status = 0;
   update_timer->start();

//#ifdef PV_USE_OPENCL
//   if((gpuAccelerateFlag)&&(true)) {
//      updateStateOpenCL(time, dt);
//   }
//   else {
//#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const PVHalo * halo = &clayer->loc.halo;

      pvdata_t * GSynHead   = GSyn[0];
//      pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//      pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
//      pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
      pvdata_t * activity = getActivity();

      switch (method) {
      case 'a':
         LIF_update_state_arma(getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), getV(), Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'b':
         LIF_update_state_beginning(getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), getV(), Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'o':
         LIF_update_state_original(getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), getV(), Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      default:
         assert(0);
         break;
      }
//#ifdef PV_USE_OPENCL
//   }
//#endif

   update_timer->stop();
   return status;
}


} // namespace PV
