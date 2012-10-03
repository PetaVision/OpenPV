/*
 * LCALifLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: slundquist
 */

#include "LCALIFLayer.hpp"
//#include "../kernels/LIF_update_state.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

namespace PV {
LCALIFLayer::LCALIFLayer() {
   initialize_base();
  // initialize(arguments) should *not* be called by the protected constructor.
}

LCALIFLayer::LCALIFLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, MAX_CHANNELS, "LCA_LIF_update_state");
}

int LCALIFLayer::initialize_base(){
   tau_LCA = 2000;
   tau_thr = 20000;
   targetRate = 1;
   integratedSpikeCount = NULL;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name){
   LIF::initialize(name, hc, TypeLCA, MAX_CHANNELS, kernel_name);
   PVParams * params = hc->parameters();
   tau_LCA = params->value(name, "tau_LCA", tau_LCA);
   tau_thr = params->value(name, "tau_thr", tau_thr);
   targetRate = params->value(name, "targetRate", targetRate);
   return PV_SUCCESS;
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount);
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   integratedSpikeCount = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(integratedSpikeCount != NULL);
   return LIF::allocateBuffers();
}

int LCALIFLayer::updateState(float time, float dt)
{
   int nk = getNumExtended();
   const pvdata_t* activityData = getLayerData();
   //Update traces
   for (int i = 0; i < nk; i++){
      integratedSpikeCount[i] += activityData[i] - (dt * integratedSpikeCount[i]/tau_LCA);
      if (i == 40){
         std::cout << activityData[i] << " " << integratedSpikeCount[i] << "\n";
      }
   }
   return LIF::updateState(time, dt);
}
/*
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
*/

}




