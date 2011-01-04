/*
 * GeislerLayer.cpp
 *
 *  Created on: Apr 21, 2010
 *      Author: gkenyon
 */

#include "GeislerLayer.hpp"
#include <assert.h>
#include <float.h>

namespace PV {

GeislerLayer::GeislerLayer(const char* name, HyPerCol * hc)
  : V1(name, hc)
{
}

GeislerLayer::GeislerLayer(const char* name, HyPerCol * hc, PVLayerType type)
  : V1(name, hc)
{
}

int GeislerLayer::updateState(float time, float dt)
{
   PVParams * params = parent->parameters();

   pv_debug_info("[%d]: V1::updateState:", clayer->columnId);

   int spikingFlag = (int) params->value(name, "spikingFlag", 1);

   assert(spikingFlag == 0);

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   pvdata_t * V = clayer->V;
   pvdata_t * phiExc  = getChannel(CHANNEL_EXC);
   pvdata_t * phiInh  = getChannel(CHANNEL_INH);
   pvdata_t * phiInhB = getChannel(CHANNEL_INHB);
   pvdata_t * activity = clayer->activity->data;

   // make sure activity in border is zero
   //
   for (int k = 0; k < clayer->numExtended; k++) {
      activity[k] = 0.0;
   }

   //pvdata_t max_bottomUp = -FLT_MAX;
   //pvdata_t max_V = -FLT_MAX;

   // assume bottomUp input to phiExc, lateral input to phiInh
   for (int k = 0; k < clayer->numNeurons; k++) {
      pvdata_t bottomUp_input = phiExc[k];
      pvdata_t lateral_input = phiInh[k];
      V[k] = (bottomUp_input > 0.0f) ? bottomUp_input * lateral_input : bottomUp_input;
      //max_bottomUp = ( max_bottomUp > bottomUp_input ) ? max_bottomUp : bottomUp_input;
      //max_V = ( max_V > V[k] ) ? max_V : V[k];
      // reset accumulation buffers
      phiExc[k] = 0.0;
      phiInh[k] = 0.0;
      phiInhB[k] = 0.0;
   }

#define SET_MAX
   //pvdata_t bottomUp_scale_factor = fabs( max_bottomUp ) / fabs( max_V + (max_V == 0.0f) );
#ifdef SET_MAX
   float max_bottomUp = 1.0f; //max_bottomUp * 2.0;
#endif
#define SET_THRESH
#ifdef SET_THRESH
   float thresh_bottomUp = 0.5f;
#endif
   for (int k = 0; k < clayer->numNeurons; k++) {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
#undef RESCALE_ACTIVITY
#ifdef RESCALE_ACTIVITY
      V[k] *= bottomUp_scale_factor;
#endif
#ifdef SET_MAX
      V[k] = V[k] > max_bottomUp ? max_bottomUp : V[k];
#endif
#ifdef SET_THRESH
      V[k] = V[k] < thresh_bottomUp ? 0.0f : V[k];
#endif
      activity[kex] = V[k];
   }

   return 0;
}

}
