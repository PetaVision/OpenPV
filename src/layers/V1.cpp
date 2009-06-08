/*
 * V1.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: rasmussn
 */

#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../connections/PVConnection.h"
#include "HyPerLayer.hpp"
#include "V1.hpp"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV
{

V1Params V1DefaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,	     // tau (ms)
    250, NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, NOISE_AMP*1.0,
    250, NOISE_AMP*1.0                       // noise (G)
};

V1::V1(const char* name, HyPerCol * hc)
{
   initialize(name, hc, TypeV1Simple);
}

V1::V1(const char* name, HyPerCol * hc, PVLayerType type)
{
   initialize(name, hc, type);
}

void V1::initialize(const char* name, HyPerCol * hc, PVLayerType type)
{
   setParent(hc);
   init(name, type);

   setParams(parent->parameters(), &V1DefaultParams);
   pvlayer_setFuncs(clayer, (INIT_FN) &LIF2_init, (UPDATE_FN) &LIF2_update_exact_linear);

   hc->addLayer(this);
}

int V1::setParams(PVParams * params, V1Params * p)
{
   const char * name = getName();
   float dt = .001 * parent->getDeltaTime();  // seconds

   clayer->params = (float *) malloc(sizeof(*p));
   memcpy(clayer->params, p, sizeof(*p));

   clayer->numParams = sizeof(*p) / sizeof(float);
   assert(clayer->numParams == 17);

   V1Params * cp = (V1Params *) clayer->params;

   if (params->present(name, "Vrest")) cp->Vrest = params->value(name, "Vrest");
   if (params->present(name, "Vexc"))  cp->Vexc  = params->value(name, "Vexc");
   if (params->present(name, "Vinh"))  cp->Vinh  = params->value(name, "Vinh");
   if (params->present(name, "VinhB")) cp->VinhB = params->value(name, "VinhB");

   if (params->present(name, "tau"))   cp->tau   = params->value(name, "tau");
   if (params->present(name, "tauE"))  cp->tauE  = params->value(name, "tauE");
   if (params->present(name, "tauI"))  cp->tauI  = params->value(name, "tauI");
   if (params->present(name, "tauIB")) cp->tauIB = params->value(name, "tauIB");

   if (params->present(name, "VthRest"))  cp->VthRest  = params->value(name, "VthRest");
   if (params->present(name, "tauVth"))   cp->tauVth   = params->value(name, "tauVth");
   if (params->present(name, "deltaVth")) cp->deltaVth = params->value(name, "deltaVth");

   if (params->present(name, "noiseAmpE"))   cp->noiseAmpE   = params->value(name, "noiseAmpE");
   if (params->present(name, "noiseAmpI"))   cp->noiseAmpI   = params->value(name, "noiseAmpI");
   if (params->present(name, "noiseAmpIB"))  cp->noiseAmpIB  = params->value(name, "noiseAmpIB");

   if (params->present(name, "noiseFreqE")) {
      cp->noiseFreqE  = params->value(name, "noiseFreqE");
      if (dt * cp->noiseFreqE > 1.0) cp->noiseFreqE = 1.0 / dt;
   }
   if (params->present(name, "noiseFreqI")) {
      cp->noiseFreqI  = params->value(name, "noiseFreqI");
      if (dt * cp->noiseFreqI > 1.0) cp->noiseFreqI = 1.0 / dt;
   }
   if (params->present(name, "noiseFreqIB")) {
      cp->noiseFreqIB = params->value(name, "noiseFreqIB");
      if (dt * cp->noiseFreqIB > 1.0) cp->noiseFreqIB = 1.0 / dt;
   }

   return 0;
}

int V1::updateState(float time, float dt)
{
   PVParams * params = parent->parameters();
   int spikingFlag = 1;

   pv_debug_info("[%d]: V1::updateState:", clayer->columnId);

   if (params->present(clayer->name, "spikingFlag")) {
      spikingFlag = params->value(clayer->name, "spikingFlag");
   }

   if (spikingFlag != 0) {
      return LIF2_update_exact_linear(clayer, dt);
   }

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   pvdata_t * V = clayer->V;
   pvdata_t * phiExc   = clayer->phi[PHI_EXC];
   pvdata_t * phiInh   = clayer->phi[PHI_INH];
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < clayer->numNeurons; k++) {
      int kPhi = kIndexExtended(k, clayer->loc.nx, clayer->loc.ny, clayer->numFeatures,
                                   clayer->numBorder);
      V[k] = phiExc[kPhi] - phiInh[kPhi];
      activity[k] = V[k];

      // reset accumulation buffers
      phiExc[kPhi] = 0.0;
      phiInh[kPhi] = 0.0;
   }

   return 0;
}

int V1::writeState(const char * path, float time)
{
   HyPerLayer::writeState(path, time);

#ifdef DEBUG_OUTPUT
   // print activity at center of image

   int sx = clayer->numFeatures;
   int sy = sx*clayer->loc.nx;
   pvdata_t * a = clayer->activity->data;

   int n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   for (int f = 0; f < clayer->numFeatures; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
   printf("\n");

   n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   n -= 8;
   for (int f = 0; f < clayer->numFeatures; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
#endif

   return 0;
}

int V1::findPostSynaptic(int dim, int maxSize, int col,
// input: which layer, which neuron
		HyPerLayer *lSource, float pos[],

		// output: how many of our neurons are connected.
		// an array with their indices.
		// an array with their feature vectors.
		int* nNeurons, int nConnectedNeurons[], float *vPos)
{
	return 0;
}

} // namespace PV
