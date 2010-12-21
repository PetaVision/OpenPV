/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: C. Rasmussen
 */

#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../connections/PVConnection.h"
#include "HyPerLayer.hpp"
#include "LIF.hpp"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV
{

LIFParams LIFDefaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,	     // tau (ms)
    250, 0*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, 0*NOISE_AMP*1.0,
    250, 0*NOISE_AMP*1.0                       // noise (G)
};

LIF::LIF(const char* name, HyPerCol * hc)
  : HyPerLayer(name, hc)
{
   initialize(TypeLIFSimple);
}

LIF::LIF(const char* name, HyPerCol * hc, PVLayerType type)
  : HyPerLayer(name, hc)
{
   initialize(type);
}

int LIF::initialize(PVLayerType type)
{
   float time = 0.0f;

   setParams(parent->parameters(), &LIFDefaultParams);

   pvlayer_setFuncs(clayer, (INIT_FN) &LIF2_init, (UPDATE_FN) &LIF2_update_exact_linear);
   this->clayer->layerType = type;

   parent->addLayer(this);

   if (parent->parameters()->value(name, "restart", 0) != 0) {
      readState(name, &time);
   }

   // initialize OpenCL parameters
   //
#ifdef PV_USE_OPENCL
   CLDevice * device = parent->getCLDevice();
   updatestate_kernel = device->createKernel("LIF_updatestate.cl", "LIF_updatestate.cl");

   // TODO - fix to use device and layer parameters
   if (device->id() == 1) {
      nxl = 1;
      nyl = 1;
   }
   else {
      nxl = 16;
      nyl = 16;
   }

   size_t lsize    = clayer->numNeurons*sizeof(pvdata_t);
   size_t lsize_ex = clayer->numExtended*sizeof(pvdata_t);

   clBuffers.V    = device->createBuffer(lsize, clayer->V);
   clBuffers.G_E  = device->createBuffer(lsize, clayer->G_E);
   clBuffers.G_I  = device->createBuffer(lsize, clayer->G_I);
   clBuffers.G_IB = device->createBuffer(lsize, clayer->G_IB);
   clBuffers.phi  = device->createBuffer(lsize, clayer->phi);
   clBuffers.activity = device->createBuffer(lsize_ex, clayer->activity);

   int argid = 0;
   updatestate_kernel->setKernelArg(argid++, clBuffers.V);
#endif

   return 0;
}

int LIF::setParams(PVParams * params,  LIFParams * p)
{
   float dt = .001 * parent->getDeltaTime();  // seconds

   clayer->params = (float *) malloc(sizeof(*p));
   assert(clayer->params != NULL);
   memcpy(clayer->params, p, sizeof(*p));

   clayer->numParams = sizeof(*p) / sizeof(float);
   assert(clayer->numParams == 17);

   LIFParams * cp = (LIFParams *) clayer->params;

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

#ifdef PV_USE_OPENCL
int LIF::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

   // setup and run kernel
   // a. unmap the state variables so device can read and write
   // b. pass state variables to kernel
   // c. run kernel
   // e. map the state variable for processing on CPU

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;

   updatestate_kernel->run(nx, ny, nxl, nyl);

   return status;
}
#endif

int LIF::updateState(float time, float dt)
{
   PVParams * params = parent->parameters();

   pv_debug_info("[%d]: LIF::updateState:", clayer->columnId);

   int spikingFlag = (int) params->value(name, "spikingFlag", 1);

   if (spikingFlag != 0) {
#ifndef PV_USE_OPENCL
      return LIF2_update_exact_linear(clayer, dt);
#else
      return updateStateOpenCL(time, dt);
#endif
   }

   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   updateV();
   setActivity();
   resetPhiBuffers();

   return 0;
}

int LIF::updateV() {
   pvdata_t * V = getV();
   pvdata_t ** phi = getCLayer()->phi;
   pvdata_t * phiExc = phi[PHI_EXC];
   pvdata_t * phiInh = phi[PHI_INH];
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] = phiExc[k] - phiInh[k];
#undef SET_MAX
#ifdef SET_MAX
      V[k] = V[k] > 1.0f ? 1.0f : V[k];
#endif
#undef SET_THRESH
#ifdef SET_THRESH
      V[k] = V[k] < 0.5f ? 0.0f : V[k];
#endif
   }
   return EXIT_SUCCESS;
}

int LIF::setActivity() {
   const int nx = getLayerLoc()->nx;
   const int ny = getLayerLoc()->ny;
   const int nf = getCLayer()->numFeatures;
   const int marginWidth = getLayerLoc()->nPad;
   pvdata_t * activity = getCLayer()->activity->data;
   pvdata_t * V = getV();
   for( int k=0; k<getNumExtended(); k++ ) {
      activity[k] = 0; // Would it be faster to only do the margins?
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      int kex = kIndexExtended( k, nx, ny, nf, marginWidth );
      activity[kex] = V[k];
   }
   return EXIT_SUCCESS;
}

int LIF::resetPhiBuffers() {
   pvdata_t ** phi = getCLayer()->phi;
   int n = getNumNeurons();
   resetBuffer( phi[PHI_EXC], n );
   resetBuffer( phi[PHI_INH], n );
   return EXIT_SUCCESS;
}

int LIF::resetBuffer( pvdata_t * buf, int numItems ) {
   for( int k=0; k<numItems; k++ ) buf[k] = 0.0;
   return EXIT_SUCCESS;
}

int LIF::writeState(const char * path, float time)
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

int LIF::findPostSynaptic(int dim, int maxSize, int col,
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
