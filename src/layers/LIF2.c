/*
 * LIF2.cpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

// ---------------------------------------------------
//  Common leaky integrate-and-fire layer routines
// ---------------------------------------------------

#include "../include/pv_common.h"
#include "PVLayer.h"

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _MSC_VER
#define inline _inline
#endif

#include "LIF2.h"

int LIF2_update_finish(PVLayer * l, float dt);

// Default handlers for a layer of leaky integrate-and-fire neurons.
static inline int update_f(PVLayer * l, int start)
{
   int k;

   LIF2_params * params = (LIF2_params *) l->params;

   /* 	const float VthRest = params->VthRest; */
   const float Vrest = params->Vrest;
   const float deltaVth = params->deltaVth;

   float * V   = l->V;
   float * Vth = l->Vth;
   float * G_E = l->G_E;
   float * G_I = l->G_I;
   float * G_IB = l->G_IB;
   float * activity = l->activity->data;

   assert(start == 0);

   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->numFeatures;
   const int marginWidth = l->loc.nPad;

   // make sure activity in border is zero
   //
   int numActive = 0;
   for (k = 0; k < l->numExtended; k++) {
      activity[k] = 0.0;
   }

   #define GMAX  10.0
   for (k = start; k < (l->numNeurons + start); k++) {
      G_E[k] = ( G_E[k] < GMAX ) ? G_E[k] : GMAX;
      G_I[k] = ( G_I[k] < GMAX ) ? G_I[k] : GMAX;
      G_IB[k] = ( G_IB[k] < GMAX ) ? G_IB[k] : GMAX;
   }

   for (k = start; k < (l->numNeurons + start); k++) {
      int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      int active = ((V[k] - Vth[k]) > 0.0) ? 1.0 : 0.0;
      activity[kex] = active;
      V[k]   -= active * (V[k] - Vrest);  // reset cells that fired
      // add hyperpolarizing current
      G_IB[k] += active * 1.0;
      Vth[k] += active * deltaVth; // reset cells that fired

      if (active) {
         // these indices are in local frame
         l->activeIndices[numActive++] = k;
      }
   }
   l->numActive = numActive;

   return 0;
}

int add_noise(PVLayer * l, float dt)
{
   int i;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;
   LIF2_params * params = (LIF2_params *) l->params;

   dt = .001 * dt;  // convert to seconds

   const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
   for (i = start; i < (l->numNeurons + start); i++) {
      if (rand() * INV_RAND_MAX < dt * params->noiseFreqE) l->phi[PHI_EXC][i]   += params->noiseAmpE * rand() * INV_RAND_MAX;
      if (rand() * INV_RAND_MAX < dt * params->noiseFreqI) l->phi[PHI_INH][i]   += params->noiseAmpI * rand() * INV_RAND_MAX;
      if (rand() * INV_RAND_MAX < dt * params->noiseFreqIB) l->phi[PHI_INHB][i] += params->noiseAmpIB * rand() * INV_RAND_MAX;
   }
   return 0;
}

int LIF2_update_explicit_euler(PVLayer * l, float dt)
{
   int i;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;
   LIF2_params * params = (LIF2_params *) l->params;

   add_noise(l, dt);

   for (i = start; i < (l->numNeurons + start); i++) {
      l->G_E[i]  += l->phi[PHI_EXC][i]  - (dt / params->tauE) * l->G_E[i];
      l->G_I[i]  += l->phi[PHI_INH][i]  - (dt / params->tauI) * l->G_I[i];
      l->G_IB[i] += l->phi[PHI_INHB][i] - (dt / params->tauIB)* l->G_IB[i];
      l->V[i] -= (dt / params->tau) *
      ( (l->V[i] - params->Vrest) +
            l->G_E[i]  * (l->V[i] - params->Vexc) +
            l->G_I[i]  * (l->V[i] - params->Vinh) +
            l->G_IB[i] * (l->V[i] - params->VinhB) );
   }
   return LIF2_update_finish(l, dt);
}

int LIF2_update_implicit_euler(PVLayer * l, float dt)
{
   int i;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;
   LIF2_params * params = (LIF2_params *) l->params;

   const float expExcite = 1. / (1. + (dt / params->tauE));
   const float expInhib  = 1. / (1. + (dt / params->tauI));
   const float expInhibB = 1. / (1. + (dt / params->tauIB));

   add_noise(l, dt);

   for (i = start; i < (l->numNeurons + start); i++)
   {
      l->G_E[i] = l->phi[PHI_EXC][i] + l->G_E[i] * expExcite;
      l->G_I[i] = l->phi[PHI_INH][i] + l->G_I[i] * expInhib;
      l->G_IB[i] = l->phi[PHI_INHB][i] + l->G_IB[i] * expInhibB;
      float Vinf = ( params->Vrest +
            l->G_E[i]  * params->Vexc +
            l->G_I[i]  * params->Vinh +
            l->G_IB[i] * params->VinhB ) /
      ( 1. + l->G_E[i] + l->G_I[i] + l->G_IB[i] );
      l->V[i] = Vinf +
      ( l->V[i] - Vinf ) /
      ( 1 + (dt / params->tau ) * ( 1 + l->G_E[i] + l->G_I[i] + l->G_IB[i] ) );
   }
   return LIF2_update_finish(l, dt);
}

int LIF2_update_exact_linear(PVLayer * l, float dt)
{
   int i;
   float tauInf, VmemInf;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;
   LIF2_params * params = (LIF2_params *) l->params;

   const float expExcite = exp(-dt / params->tauE );
   const float expInhib  = exp(-dt / params->tauI );
   const float expInhibB = exp(-dt / params->tauIB);

   const float Vrest = params->Vrest;
   const float Vexc  = params->Vexc;
   const float Vinh  = params->Vinh;
   const float VinhB = params->VinhB;
   const float tau   = params->tau;

   float * V = l->V;

   float * G_E  = l->G_E;
   float * G_I  = l->G_I;
   float * G_IB = l->G_IB;

   float * phiExc  = l->phi[PHI_EXC];
   float * phiInh  = l->phi[PHI_INH];
   float * phiInhB = l->phi[PHI_INHB];

   add_noise(l, dt);

   for (i = start; i < (l->numNeurons + start); i++)
   {
      G_E[i]  = phiExc[i]  + G_E[i]  * expExcite;
      G_I[i]  = phiInh[i]  + G_I[i]  * expInhib;
      G_IB[i] = phiInhB[i] + G_IB[i] * expInhibB;
      tauInf  = (dt / tau) * (1 + G_E[i] + G_I[i] + G_IB[i]);
      VmemInf = ( Vrest + G_E[i] * Vexc + G_I[i] * Vinh + G_IB[i] * VinhB )
                / (1.0 + G_E[i] + G_I[i] + G_IB[i]);
//      tauInf  = (dt / tau) * (1 + G_E[i] + G_I[i]);
//      VmemInf = ( Vrest + G_E[i] * Vexc + G_I[i] * Vinh )
//                / (1.0 + G_E[i] + G_I[i]);
      V[i] = VmemInf + (V[i] - VmemInf) * exp(-tauInf);
   }

   return LIF2_update_finish(l, dt);
}

void print_stats(PVLayer * l)
{
   float phiAve  = 0.0, phiMax  = FLT_MIN, phiMin  = FLT_MAX;
   float phi1Ave = 0.0, phi1Max = FLT_MIN, phi1Min = FLT_MAX;
   float phi2Ave = 0.0, phi2Max = FLT_MIN, phi2Min = FLT_MAX;
   float GexcAve = 0.0, GexcMax = FLT_MIN, GexcMin = FLT_MAX;
   float GinhAve = 0.0, GinhMax = FLT_MIN, GinhMin = FLT_MAX;
   float GinhBAve = 0.0, GinhBMax = FLT_MIN, GinhBMin = FLT_MAX;
   float VAve = 0.0, VMax = -FLT_MAX, VMin = FLT_MAX;
   float VthAve = 0.0, VthMax = -FLT_MAX, VthMin = FLT_MAX;
   char msg[128];
   int i;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;

   for (i = start; i < (l->numNeurons + start); i++) {
      // Gather some statistics
      phiAve += l->phi[PHI_EXC][i];
      if (l->phi[PHI_EXC][i] < phiMin) phiMin = l->phi[PHI_EXC][i];
      if (l->phi[PHI_EXC][i] > phiMax) phiMax = l->phi[PHI_EXC][i];

      phi1Ave += l->phi[PHI_INH][i];
      if (l->phi[PHI_INH][i] < phi1Min) phi1Min = l->phi[PHI_INH][i];
      if (l->phi[PHI_INH][i] > phi1Max) phi1Max = l->phi[PHI_INH][i];

      phi2Ave += l->phi[PHI_INHB][i];
      if (l->phi[PHI_INHB][i] < phi2Min) phi2Min = l->phi[PHI_INHB][i];
      if (l->phi[PHI_INHB][i] > phi2Max) phi2Max = l->phi[PHI_INHB][i];

      GexcAve += l->G_E[i];
      if (l->G_E[i] < GexcMin) GexcMin = l->G_E[i];
      if (l->G_E[i] > GexcMax) GexcMax = l->G_E[i];

      GinhAve += l->G_I[i];
      if (l->G_I[i] < GinhMin) GinhMin = l->G_I[i];
      if (l->G_I[i] > GinhMax) GinhMax = l->G_I[i];

      GinhBAve += l->G_IB[i];
      if (l->G_IB[i] < GinhBMin) GinhBMin = l->G_IB[i];
      if (l->G_IB[i] > GinhBMax) GinhBMax = l->G_IB[i];

      VAve += l->V[i];
      if (l->V[i] < VMin) VMin = l->V[i];
      if (l->V[i] > VMax) VMax = l->V[i];

      VthAve += l->Vth[i];
      if (l->Vth[i] < VthMin) VthMin = l->Vth[i];
      if (l->Vth[i] > VthMax) VthMax = l->Vth[i];
   }

   if (0) {  // TODO - fix threads
   //   if (l->yOrigin == 0) {
      sprintf(msg, "[0]: L%d: phi0: Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            phiMax, phiAve / l->numNeurons, phiMin);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: phi1: Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            phi1Max, phi1Ave / l->numNeurons, phi1Min);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: phi2: Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            phi2Max, phi2Ave / l->numNeurons, phi2Min);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: G_E : Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            GexcMax, GexcAve / l->numNeurons, GexcMin);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: G_I : Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            GinhMax, GinhAve / l->numNeurons, GinhMin);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: G_IB: Max:   %1.4f, Avg=  %1.4f Min=  %1.4f\n", l->layerId,
            GinhBMax, GinhBAve / l->numNeurons, GinhBMin);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: V   : Max: %1.4f, Avg=%1.4f Min=%1.4f\n", l->layerId,
            VMax, VAve / l->numNeurons, VMin);
      pv_log(stdout, msg);
      sprintf(msg, "[0]: L%d: Vth : Max: %1.4f, Avg=%1.4f Min=%1.4f\n", l->layerId,
            VthMax, VthAve / l->numNeurons, VthMin);
      pv_log(stdout, msg);
   }

} //  layer_stats

int LIF2_update_finish(PVLayer * l, float dt)
{
   int i;
   int start = 0;  // TODO - fix for threads
   //   int start = (l->yOrigin * l->loc.nx + l->xOrigin) * l->numFeatures;
   LIF2_params * params = (LIF2_params *) l->params;

   const float VthRest = params->VthRest;
   const float expVth  = exp(-dt / params->tauVth);
   print_stats(l);

   for (i = start; i < (l->numNeurons + start); i++)
   {
      l->phi[PHI_EXC][i]  = 0.0;
      l->phi[PHI_INH][i]  = 0.0;
      l->phi[PHI_INHB][i] = 0.0;
      l->Vth[i] = VthRest + (l->Vth[i] - VthRest) * expVth;
   }

   return update_f(l, start); // resets V if f == 1
}

int LIF2_init(PVLayer * l)
{
   int k, m, kex;
   LIF2_params * params = (LIF2_params *) l->params;

   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->numFeatures;
   const int marginWidth = l->loc.nPad;

   for (k = 0; k < l->numNeurons; k++) {
      for (m = 0; m < l->numPhis; m++) {
         l->phi[m][k] = 0.0;
      }

      kex = kIndexExtended(k, nx, ny, nf, marginWidth);

      if (params->noiseAmpE > 0) {
         l->Vth[k] = params->VthRest + (rand() / (float) RAND_MAX) * params->deltaVth;
         l->V[k] = (rand() / (float) RAND_MAX) * (params->VthRest - params->Vinh) + params->Vinh;
         if ( (l->layerType == TypeV1Simple) || (l->layerType == TypeV1Simple2) ) {
            l->G_E[k] = -log( rand() / (float) RAND_MAX ) * 0.6860;
            l->G_I[k] = -log( rand() / (float) RAND_MAX ) * 1.0031;
            l->G_IB[k] = -log( rand() / (float) RAND_MAX ) * 5.5048;
         }
         else if (l->layerType == TypeV1FlankInhib) {
            l->G_E[k] = -log( rand() / (float) RAND_MAX ) * 0.4514;
            l->G_I[k] = -log( rand() / (float) RAND_MAX ) * 3.2228;
            l->G_IB[k] = -log( rand() / (float) RAND_MAX ) * 5.8198;
         }
         else if ( (l->layerType == TypeV1FeedbackInhib) || (l->layerType == TypeV1Feedback2Inhib) ) {
            l->G_E[k] = -log( rand() / (float) RAND_MAX ) * 0.3932;
            l->G_I[k] = -log( rand() / (float) RAND_MAX ) * 0.7572;
            l->G_IB[k] = -log( rand() / (float) RAND_MAX ) * 1.8671;
         }
         else if (l->layerType == TypeV1SurroundInhib) {
            l->G_E[k] = -log( rand() / (float) RAND_MAX ) * 0.4159;
            l->G_I[k] = -log( rand() / (float) RAND_MAX ) * 0.4956;
            l->G_IB[k] = -log( rand() / (float) RAND_MAX ) * 0.9782;
         }
         else {
            l->G_E[k] = -log( rand() / (float) RAND_MAX );
            l->G_I[k] = -log( rand() / (float) RAND_MAX );
            l->G_IB[k] = -log( rand() / (float) RAND_MAX );
         }
      }
      else
      {
         l->Vth[k] = params->VthRest;
         l->V[k]   = params->Vrest;
      }

      // TODO - Initialize activity buffers with random noise
      // TODO - use a parameter for RAND threshold

      if (params->noiseAmpE > 0) {
         if ( (l->layerType == TypeV1Simple) || (l->layerType == TypeV1Simple2) ) {
            l->activity->data[kex] = (float) (rand() / (float) RAND_MAX)
            < 0.001;
         }
         else if (l->layerType == TypeV1FlankInhib) {
            l->activity->data[kex] = (float) (rand() / (float) RAND_MAX)
            < 0.001;
         }
         else if ( (l->layerType == TypeV1FeedbackInhib) || (l->layerType == TypeV1Feedback2Inhib) ) {
            l->activity->data[kex] = (float) (rand() / (float) RAND_MAX)
            < 0.0016;
         }
         else if (l->layerType == TypeV1SurroundInhib) {
            l->activity->data[kex] = (float) (rand() / (float) RAND_MAX)
            < 0.01;
         }
         else {
            l->activity->data[kex] = 0.0;
         }
      }
      else
         l->activity->data[kex] = 0.0;

   }

   if ( (params->noiseAmpE == 0.0) && (l->layerType == -TypeV1Simple) ) {
      int i_theta = 0;
      /* 		int i_kurve = 0; */
      /* 		int k = (int) ((l->ny / 2.0) * l->loc.nx * NO * NK + (l->loc.nx / 2.0) * NO */
      /* 				* NK + i_theta * NK + i_kurve); */
      int k = (int) ((l->loc.ny / 2.0) * l->loc.nx * NO + (l->loc.nx / 2.0) * NO
            + i_theta);
      kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      l->activity->data[kex] = 1.0;
   }
   else if ( (params->noiseAmpE == 0.0) && (l->layerType == TypeV1FlankInhib) ) {
      int i_theta = 0;
      int k = (int) ((l->loc.ny / 2.0) * l->loc.nx * NO + (l->loc.nx / 2.0) * NO
            + i_theta);
      kex = kIndexExtended(k, nx, ny, nf, marginWidth);
      l->activity->data[kex] = 1.0;
   }
   else if (l->layerType == -TypeV1SurroundInhib)
   {
      assert(0 == 1);  // TODO - fix this code (never used?)
      /* int i_theta = 5; */
      /* int k = (int) ((l->ny / 2.0) * l->loc.nx * NO + (l->loc.nx / 2.0) * NO */
      /* 	+ i_theta); */
      /* l->activity->data[MAX_F_DELAY - 1][k] = 1.0; */
      l->activity->data[((NY / 2) - 2) * NX + ( (NX / 2) - 2) ] = 1.0;
      l->activity->data[((NY / 2) - 1) * NX + ( (NX / 2) - 1) ] = 1.0;
      l->activity->data[((NY / 2) - 0) * NX + ( (NX / 2) - 0) ] = 1.0;
      l->activity->data[((NY / 2) + 1) * NX + ( (NX / 2) + 1) ] = 1.0;
      l->activity->data[((NY / 2) + 2) * NX + ( (NX / 2) + 2) ] = 1.0;
   }

   return 0;
}
