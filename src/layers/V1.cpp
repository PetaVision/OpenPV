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

LIFParams LIFDefaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,	     // tau (ms)
    250, 0*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, 0*NOISE_AMP*1.0,
    250, 0*NOISE_AMP*1.0                       // noise (G)
};

V1::V1(const char* name, HyPerCol * hc)
  : HyPerLayer(name, hc)
{
   initialize(TypeLIFSimple);
}

V1::V1(const char* name, HyPerCol * hc, PVLayerType type)
  : HyPerLayer(name, hc)
{
   initialize(type);
}

int V1::initialize(PVLayerType type)
{
   float time = 0.0f;
   int status = CL_SUCCESS;

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
#endif

   return status;
}

#ifdef PV_USE_OPENCL
int V1::initializeThreadData()
{
   int status = CL_SUCCESS;

   // map layer buffers so that layer data can be initialized
   //
   pvdata_t * V = (pvdata_t *)   clBuffers.V->map(CL_MAP_WRITE);
   pvdata_t * Vth = (pvdata_t *) clBuffers.Vth->map(CL_MAP_WRITE);

   // initialize layer data
   //
   for (int k = 0; k < clayer->numNeurons; k++){
      V[k] = V_REST;
   }

   for (int k = 0; k < clayer->numNeurons; k++){
      Vth[k] = VTH_REST;
   }

   clBuffers.V->unmap(V);
   clBuffers.Vth->unmap(Vth);

   return status;
}

int V1::initializeThreadKernels()
{
   int status = CL_SUCCESS;

   // create kernels
   //
   updatestate_kernel = parent->getCLDevice()->createKernel("LIF_updatestate.cl", "update_state");

   int argid = 0;
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.V);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_E);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_I);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.G_IB);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.phi);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.activity);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nx);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.ny);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->numFeatures);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nPad);

   return status;
}
#endif

int V1::setParams(PVParams * params, LIFParams * p)
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
int V1::updateStateOpenCL(float time, float dt)
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

int V1::updateState(float time, float dt)
{
   PVParams * params = parent->parameters();

   pv_debug_info("[%d]: V1::updateState:", clayer->columnId);

   int spikingFlag = (int) params->value(name, "spikingFlag", 1);

   if (spikingFlag != 0) {
#ifndef PV_USE_OPENCL
      return LIF2_update_exact_linear(clayer, dt);
#else
      return updateStateOpenCL(time, dt);
#endif
   }
   else {
      return HyPerLayer::updateState(time, dt);
   }
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
