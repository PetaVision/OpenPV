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
   const int nf = clayer->numFeatures;
   const int marginWidth = clayer->loc.nPad;

   pvdata_t * V = clayer->V;
   pvdata_t * phiExc   = clayer->phi[PHI_EXC];
   pvdata_t * phiInh  = clayer->phi[PHI_INH];
   pvdata_t * phiInhB   = clayer->phi[PHI_INHB];
   pvdata_t * activity = clayer->activity->data;

   // make sure activity in border is zero
   //
   for (int k = 0; k < clayer->numExtended; k++) {
      activity[k] = 0.0;
   }

   pvdata_t max_direct = -FLT_MAX;
   pvdata_t max_V = -FLT_MAX;

   // assume direct input to phiExc, lateral input to phiInh
   for (int k = 0; k < clayer->numNeurons; k++) {
      pvdata_t direct_input = phiExc[k];
      pvdata_t lateral_input = phiInh[k];
      V[k] = (direct_input > 0.0f) ? direct_input * lateral_input : direct_input;
      max_direct = ( max_direct > direct_input ) ? max_direct : direct_input;
      max_V = ( max_V > V[k] ) ? max_V : V[k];
      // reset accumulation buffers
      phiExc[k] = 0.0;
      phiInh[k] = 0.0;
      phiInhB[k] = 0.0;
   }

   pvdata_t scale_factor = fabs( max_direct ) / fabs( max_V + (max_V == 0.0f) );
   for (int k = 0; k < clayer->numNeurons; k++) {
      int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      //V[k] *= scale_factor;
      activity[kex] = ( V[k] > 0.0f ) ? V[k] : 0.0f;
   }

   return 0;
}

}
