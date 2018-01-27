/*
 * LCALIFLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: slundquist & dpaiton
 */

#include "LCALIFLayer.hpp"

#include <cmath>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils/cl_random.h"

//////////////////////////////////////////////////////
// implementation of LIF kernels
//////////////////////////////////////////////////////

// Kernel update state implementation receives all necessary variables
// for required computation. File is included above.
void LCALIF_update_state(
      const int nbatch,
      const int numNeurons,
      const double timed,
      const double dt,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float Vscale,
      float *Vadpt,
      const float tauTHR,
      const float targetRateHz,

      float *integratedSpikeCount,

      LIF_params *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float *GSynHead,
      float *activity,

      const float *gapStrength,
      float *Vattained,
      float *Vmeminf,
      const int normalizeInputFlag,
      float *GSynExcEffective,
      float *GSynInhEffective,
      float *excitatoryNoise,
      float *inhibitoryNoise,
      float *inhibNoiseB);

namespace PV {
LCALIFLayer::LCALIFLayer() {
   initialize_base();
   // initialize(arguments) should *not* be called by the protected constructor.
}

LCALIFLayer::LCALIFLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc, "LCALIF_update_state");
}

int LCALIFLayer::initialize_base() {
   numChannels          = 5;
   tauTHR               = 1000;
   targetRateHz         = 1;
   Vscale               = DEFAULT_DYNVTHSCALE;
   Vadpt                = NULL;
   integratedSpikeCount = NULL;
   G_Norm               = NULL;
   GSynExcEffective     = NULL;
   normalizeInputFlag   = false;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char *name, HyPerCol *hc, const char *kernel_name) {
   LIFGap::initialize(name, hc, kernel_name);

   float defaultDynVthScale = lParams.VthRest - lParams.Vrest;
   Vscale                   = defaultDynVthScale > 0 ? defaultDynVthScale : DEFAULT_DYNVTHSCALE;
   if (Vscale <= 0) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "LCALIFLayer \"%s\": Vscale must be positive (value in params is %f).\n",
               name,
               (double)Vscale);
      }
      abort();
   }

   return PV_SUCCESS;
}

int LCALIFLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LIFGap::ioParamsFillGroup(ioFlag);
   ioParam_tauTHR(ioFlag);
   ioParam_targetRate(ioFlag);
   ioParam_normalizeInput(ioFlag);
   ioParam_Vscale(ioFlag);
   return status;
}

void LCALIFLayer::ioParam_tauTHR(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "tauTHR", &tauTHR, tauTHR);
}

void LCALIFLayer::ioParam_targetRate(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "targetRate", &targetRateHz, targetRateHz);
}

void LCALIFLayer::ioParam_normalizeInput(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "normalizeInput", &normalizeInputFlag, normalizeInputFlag);
}

void LCALIFLayer::ioParam_Vscale(enum ParamsIOFlag ioFlag) {
   PVParams *p = parent->parameters();
   assert(!p->presentAndNotBeenRead(name, "VthRest"));
   assert(!p->presentAndNotBeenRead(name, "Vrest"));
   float defaultDynVthScale = lParams.VthRest - lParams.Vrest;
   Vscale                   = defaultDynVthScale > 0 ? defaultDynVthScale : DEFAULT_DYNVTHSCALE;
   parent->parameters()->ioParamValue(ioFlag, name, "Vscale", &Vscale, Vscale);
}

LCALIFLayer::~LCALIFLayer() {
   free(integratedSpikeCount);
   integratedSpikeCount = NULL;
   free(Vadpt);
   Vadpt = NULL;
   free(Vattained);
   Vattained = NULL;
   free(Vmeminf);
   Vmeminf = NULL;
}

Response::Status LCALIFLayer::allocateDataStructures() {
   auto status = LIFGap::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   int numNeurons = getNumNeuronsAllBatches();
   for (int k = 0; k < numNeurons; k++) {
      integratedSpikeCount[k] =
            targetRateHz / 1000; // Initialize integrated spikes to non-zero value
      Vadpt[k]     = lParams.VthRest; // Initialize integrated spikes to non-zero value
      Vattained[k] = lParams.Vrest;
      Vmeminf[k]   = lParams.Vrest;
   }

   return Response::SUCCESS;
}

void LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeuronsAllBatches();
   // Allocate data to keep track of trace
   allocateBuffer(&integratedSpikeCount, numNeurons, "integratedSpikeCount");
   allocateBuffer(&Vadpt, numNeurons, "Vadpt");
   allocateBuffer(&Vattained, numNeurons, "Vattained");
   allocateBuffer(&Vmeminf, numNeurons, "Vmeminf");
   allocateBuffer(&G_Norm, numNeurons, "G_Norm");
   allocateBuffer(&GSynExcEffective, numNeurons, "GSynExcEffective");
   allocateBuffer(&GSynInhEffective, numNeurons, "GSynInhEffective");
   allocateBuffer(&excitatoryNoise, numNeurons, "excitatoryNoise");
   allocateBuffer(&inhibitoryNoise, numNeurons, "inhibitoryNoise");
   allocateBuffer(&inhibNoiseB, numNeurons, "inhibNoiseB");
   LIFGap::allocateBuffers();
}

Response::Status LCALIFLayer::registerData(Checkpointer *checkpointer) {
   auto status = LIFGap::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   checkpointPvpActivityFloat(
         checkpointer, "integratedSpikeCount", integratedSpikeCount, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "Vadpt", Vadpt, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "Vattained", Vattained, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "Vmeminf", Vmeminf, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "G_Norm", G_Norm, false /*not extended*/);
   checkpointPvpActivityFloat(
         checkpointer, "GSynExcEffective", GSynExcEffective, false /*not extended*/);
   checkpointPvpActivityFloat(
         checkpointer, "GSynInhEffective", GSynInhEffective, false /*not extended*/);
   checkpointPvpActivityFloat(
         checkpointer, "excitatoryNoise", excitatoryNoise, false /*not extended*/);
   checkpointPvpActivityFloat(
         checkpointer, "inhibitoryNoise", inhibitoryNoise, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "inhibNoiseB", inhibNoiseB, false /*not extended*/);
   return Response::SUCCESS;
}

Response::Status LCALIFLayer::updateState(double timed, double dt) {
   // Calculate_state kernel
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      G_Norm[k] = GSyn[CHANNEL_NORM][k]; // Copy GSyn buffer on normalizing channel for
      // checkpointing, since LCALIF_update_state will blank the
      // GSyn's
   }
   LCALIF_update_state(
         clayer->loc.nbatch,
         getNumNeurons(),
         timed,
         dt,
         clayer->loc.nx,
         clayer->loc.ny,
         clayer->loc.nf,
         clayer->loc.halo.lt,
         clayer->loc.halo.rt,
         clayer->loc.halo.dn,
         clayer->loc.halo.up,
         Vscale,
         Vadpt,
         tauTHR,
         targetRateHz,
         integratedSpikeCount,
         &lParams,
         randState->getRNG(0),
         clayer->V,
         Vth,
         G_E,
         G_I,
         G_IB,
         GSyn[0],
         clayer->activity->data,
         getGapStrength(),
         Vattained,
         Vmeminf,
         (int)normalizeInputFlag,
         GSynExcEffective,
         GSynInhEffective,
         excitatoryNoise,
         inhibitoryNoise,
         inhibNoiseB);
   return Response::SUCCESS;
}

Response::Status LCALIFLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (initializeFromCheckpointFlag) {
      auto status = LIFGap::readStateFromCheckpoint(checkpointer);
      if (status != Response::SUCCESS) {
         return status;
      }
      read_integratedSpikeCountFromCheckpoint(checkpointer);
      readVadptFromCheckpoint(checkpointer);
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void LCALIFLayer::read_integratedSpikeCountFromCheckpoint(Checkpointer *checkpointer) {
   std::string checkpointEntryName(name);
   checkpointEntryName.append("_integratedSpikeCount.pvp");
}

void LCALIFLayer::readVadptFromCheckpoint(Checkpointer *checkpointer) {
   std::string checkpointEntryName(name);
   checkpointEntryName.append("_Vadpt.pvp");
}

} // namespace PV

// Kernels

inline float LCALIF_tauInf(
      const float tau,
      const float G_E,
      const float G_I,
      const float G_IB,
      const float sum_gap) {
   return tau / (1 + G_E + G_I + G_IB + sum_gap);
}

inline float LCALIF_VmemInf(
      const float Vrest,
      const float V_E,
      const float V_I,
      const float V_B,
      const float G_E,
      const float G_I,
      const float G_B,
      const float G_gap,
      const float sumgap) {
   return (Vrest + V_E * G_E + V_I * G_I + V_B * G_B + G_gap) / (1 + G_E + G_I + G_B + sumgap);
}

//
// update the state of a LCALIF layer (spiking)
//
//    assume called with 1D kernel
//

void LCALIF_update_state(
      const int nbatch,
      const int numNeurons,
      const double timed,
      const double dt,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float Vscale,
      float *Vadpt,
      float tauTHR,
      const float targetRateHz,

      float *integratedSpikeCount,

      LIF_params *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float *GSynHead,
      float *activity,

      const float *gapStrength,
      float *Vattained,
      float *Vmeminf,
      const int normalizeInputFlag,
      float *GSynExcEffective,
      float *GSynInhEffective,
      float *excitatoryNoise,
      float *inhibitoryNoise,
      float *inhibNoiseB) {

   // convert target rate from Hz to kHz
   float targetRatekHz = targetRateHz / 1000.0f;

   // tau parameters
   const float tauO = 1 / targetRatekHz; // Convert target rate from kHz to ms (tauO)

   const float decayE   = expf((float)-dt / params->tauE);
   const float decayI   = expf((float)-dt / params->tauI);
   const float decayIB  = expf((float)-dt / params->tauIB);
   const float decayVth = expf((float)-dt / params->tauVth);
   const float decayO   = expf((float)-dt / tauO);

   // Convert dt to seconds
   const float dt_sec = (float)(0.001 * dt);

   for (int k = 0; k < nx * ny * nf * nbatch; k++) {
      int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //

      // local param variables
      float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

      // local variables
      float l_activ;

      taus_uint4 l_rnd = rnd[k];

      float l_V   = V[k];
      float l_Vth = Vth[k];

      // The correction factors to the conductances are so that if l_GSyn_* is the same every
      // timestep,
      // then the asymptotic value of l_G_* will be l_GSyn_*
      float l_G_E         = G_E[k];
      float l_G_I         = G_I[k];
      float l_G_IB        = G_IB[k];
      float l_gapStrength = gapStrength[k];

#define CHANNEL_NORM (CHANNEL_GAP + 1)
      float *GSynExc   = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
      float *GSynInh   = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
      float *GSynInhB  = &GSynHead[CHANNEL_INHB * nbatch * numNeurons];
      float *GSynGap   = &GSynHead[CHANNEL_GAP * nbatch * numNeurons];
      float *GSynNorm  = &GSynHead[CHANNEL_NORM * nbatch * numNeurons];
      float l_GSynExc  = GSynExc[k];
      float l_GSynInh  = GSynInh[k];
      float l_GSynInhB = GSynInhB[k];
      float l_GSynGap  = GSynGap[k];
      float l_GSynNorm = normalizeInputFlag ? GSynNorm[k] : 1.0f;

      // define local param variables
      //
      tau   = params->tau;
      Vexc  = params->Vexc;
      Vinh  = params->Vinh;
      VinhB = params->VinhB;
      Vrest = params->Vrest;

      VthRest  = params->VthRest;
      deltaVth = params->deltaVth;
      deltaGIB = params->deltaGIB;

      if (normalizeInputFlag && l_GSynNorm == 0 && l_GSynExc != 0) {
         ErrorLog().printf(
               "time = %f, k = %d, normalizeInputFlag is true but GSynNorm is zero and l_GSynExc = "
               "%f\n",
               timed,
               k,
               (double)l_GSynExc);
         abort();
      }
      l_GSynExc /= (l_GSynNorm + (l_GSynNorm == 0 ? 1 : 0));
      GSynExcEffective[k] = l_GSynExc;
      GSynInhEffective[k] = l_GSynInh;

      // add noise
      //
      excitatoryNoise[k] = 0.0f;
      l_rnd              = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqE) {
         l_rnd              = cl_random_get(l_rnd);
         excitatoryNoise[k] = params->noiseAmpE * cl_random_prob(l_rnd);
         l_GSynExc          = l_GSynExc + excitatoryNoise[k];
      }

      inhibitoryNoise[k] = 0.0f;
      l_rnd              = cl_random_get(l_rnd);
      float r            = cl_random_prob(l_rnd);
      if (r < dt_sec * params->noiseFreqI) {
         l_rnd              = cl_random_get(l_rnd);
         r                  = cl_random_prob(l_rnd);
         inhibitoryNoise[k] = params->noiseAmpI * r;
         l_GSynInh          = l_GSynInh + inhibitoryNoise[k];
      }

      inhibNoiseB[k] = 0.0f;
      l_rnd          = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqIB) {
         l_rnd          = cl_random_get(l_rnd);
         inhibNoiseB[k] = params->noiseAmpIB * cl_random_prob(l_rnd);
         l_GSynInhB     = l_GSynInhB + inhibNoiseB[k];
      }

      const float GMAX = FLT_MAX;

      // The portion of code below uses the newer method of calculating l_V.
      float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
      float tau_inf_initial, tau_inf_final, V_inf_initial, V_inf_final;

      G_E_initial     = l_G_E + l_GSynExc;
      G_I_initial     = l_G_I + l_GSynInh;
      G_IB_initial    = l_G_IB + l_GSynInhB;
      tau_inf_initial = LCALIF_tauInf(tau, G_E_initial, G_I_initial, G_IB_initial, l_gapStrength);

      V_inf_initial = LCALIF_VmemInf(
            Vrest,
            Vexc,
            Vinh,
            VinhB,
            G_E_initial,
            G_I_initial,
            G_IB_initial,
            l_GSynGap,
            l_gapStrength);

      G_E_initial  = (G_E_initial > GMAX) ? GMAX : G_E_initial;
      G_I_initial  = (G_I_initial > GMAX) ? GMAX : G_I_initial;
      G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

      float totalconductance = 1.0f + G_E_initial + G_I_initial + G_IB_initial + l_gapStrength;
      Vmeminf[k] =
            (Vrest + Vexc * G_E_initial + Vinh * G_I_initial + VinhB * G_IB_initial + l_GSynGap)
            / totalconductance;

      G_E_final     = G_E_initial * decayE;
      G_I_final     = G_I_initial * decayI;
      G_IB_final    = G_IB_initial * decayIB;
      tau_inf_final = LCALIF_tauInf(tau, G_E_final, G_I_initial, G_IB_initial, l_gapStrength);
      V_inf_final   = LCALIF_VmemInf(
            Vrest, Vexc, Vinh, VinhB, G_E_final, G_I_final, G_IB_final, l_GSynGap, l_gapStrength);

      float tau_slope = (tau_inf_final - tau_inf_initial) / (float)dt;
      float f1        = tau_slope == 0.0f ? expf(-(float)dt / tau_inf_initial)
                                   : powf(tau_inf_final / tau_inf_initial, -1.0f / tau_slope);
      float f2 = tau_slope == -1.0f
                       ? tau_inf_initial / (float)dt * logf(tau_inf_final / tau_inf_initial + 1.0f)
                       : (1.0f - tau_inf_initial / (float)dt * (1.0f - f1)) / (1.0f + tau_slope);
      float f3 = 1.0f - f1 - f2;
      l_V      = f1 * l_V + f2 * V_inf_initial + f3 * V_inf_final;

      l_G_E  = G_E_final;
      l_G_I  = G_I_final;
      l_G_IB = G_IB_final;

      // l_Vth updates according to traditional LIF rule in addition to the slow threshold
      // adaptation
      //      See LCA_Equations.pdf in the documentation for a full description of the neuron
      //      adaptive firing threshold.

      Vadpt[k] = -60.0f;

      l_Vth = Vadpt[k] + decayVth * (l_Vth - Vadpt[k]);

      bool fired_flag = (l_V > l_Vth);

      l_activ      = fired_flag ? 1.0f : 0.0f;
      Vattained[k] = l_V; // Save the value of V before it drops due to the spike
      l_V          = fired_flag ? Vrest : l_V;
      l_Vth        = fired_flag ? l_Vth + deltaVth : l_Vth;
      l_G_IB       = fired_flag ? l_G_IB + deltaGIB : l_G_IB;

      // integratedSpikeCount is the trace activity of the neuron, with an exponential decay
      integratedSpikeCount[k] = decayO * (l_activ + integratedSpikeCount[k]);

      //
      // These actions must be done outside of kernel
      //    1. set activity to 0 in boundary (if needed)
      //    2. update active indices
      //

      // store local variables back to global memory
      //
      rnd[k] = l_rnd;

      activity[kex] = l_activ;

      V[k]   = l_V;
      Vth[k] = l_Vth;

      G_E[k]  = l_G_E;
      G_I[k]  = l_G_I;
      G_IB[k] = l_G_IB;

   } // loop over k
}
