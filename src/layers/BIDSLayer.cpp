/*
 * BIDSLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 */

#include "HyPerLayer.hpp"
#include "BIDSLayer.hpp"
#include "LIF.hpp"
#include "../kernels/LIF_update_state.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

namespace PV {
BIDSLayer::BIDSLayer() {
  // initialize(arguments) should *not* be called by the protected constructor.
}

BIDSLayer::BIDSLayer(const char * name, HyPerCol * hc) {
   initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
}

int BIDSLayer::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name){
   LIF::initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
   return PV_SUCCESS;
}

int BIDSLayer::updateState(float time, float dt)
{
   int status = 0;
   update_timer->start();

#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(true)) {
      updateStateOpenCL(time, dt);
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
//      pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
      pvdata_t * activity = clayer->activity->data;

      switch (method) {
      case 'b':
         LIF_update_state_beginning(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'o':
         LIF_update_state_original(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      default:
         assert(0);
         break;
      }
#ifdef PV_USE_OPENCL
   }
#endif

   updateActiveIndices();
   update_timer->stop();
   return status;
}


}
