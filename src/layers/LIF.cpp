/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "LIF.hpp"
#include "HyPerLayer.hpp"

#include "checkpointing/CheckpointEntryRandState.hpp"
#include "components/LIFLayerInputBuffer.hpp"
#include "include/default_params.h"
#include "include/pv_common.h"
#include "io/fileio.hpp"
#include "io/randomstateio.hpp"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

void LIF_update_state_arma(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,

      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity);

void LIF_update_state_beginning(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,

      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity);

void LIF_update_state_original(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,

      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity);

namespace PV {

LIF::LIF() { initialize_base(); }

LIF::LIF(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc, "LIF_update_state");
}

LIF::~LIF() {
   free(Vth);
   delete randState;
   free(methodString);
}

int LIF::initialize_base() {
   randState    = NULL;
   Vth          = NULL;
   G_E          = NULL;
   G_I          = NULL;
   G_IB         = NULL;
   methodString = NULL;

   return PV_SUCCESS;
}

int LIF::initialize(const char *name, HyPerCol *hc, const char *kernel_name) {
   HyPerLayer::initialize(name, hc);
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);
   mLayerInput->requireChannel(CHANNEL_INHB);
   return PV_SUCCESS;
}

int LIF::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   HyPerLayer::ioParamsFillGroup(ioFlag);

   ioParam_Vrest(ioFlag);
   ioParam_Vexc(ioFlag);
   ioParam_Vinh(ioFlag);
   ioParam_VinhB(ioFlag);
   ioParam_VthRest(ioFlag);
   ioParam_tau(ioFlag);
   ioParam_tauVth(ioFlag);
   ioParam_deltaVth(ioFlag);
   ioParam_deltaGIB(ioFlag);

   // NOTE: in LIFDefaultParams, noise ampE, ampI, ampIB were
   // ampE=0*NOISE_AMP*( 1.0/TAU_EXC )
   //       *(( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST))
   // ampI=0*NOISE_AMP*1.0
   // ampIB=0*NOISE_AMP*1.0
   //

   ioParam_noiseAmpE(ioFlag);
   ioParam_noiseAmpI(ioFlag);
   ioParam_noiseAmpIB(ioFlag);
   ioParam_noiseFreqE(ioFlag);
   ioParam_noiseFreqI(ioFlag);
   ioParam_noiseFreqIB(ioFlag);

   ioParam_method(ioFlag);
   return 0;
}
void LIF::ioParam_Vrest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vrest", &lParams.Vrest, (float)V_REST);
}
void LIF::ioParam_Vexc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vexc", &lParams.Vexc, (float)V_EXC);
}
void LIF::ioParam_Vinh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vinh", &lParams.Vinh, (float)V_INH);
}
void LIF::ioParam_VinhB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VinhB", &lParams.VinhB, (float)V_INHB);
}
void LIF::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VthRest", &lParams.VthRest, (float)VTH_REST);
}
void LIF::ioParam_tau(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tau", &lParams.tau, (float)TAU_VMEM);
}
void LIF::ioParam_tauVth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauVth", &lParams.tauVth, (float)TAU_VTH);
}
void LIF::ioParam_deltaVth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "deltaVth", &lParams.deltaVth, (float)DELTA_VTH);
}
void LIF::ioParam_deltaGIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "deltaGIB", &lParams.deltaGIB, (float)DELTA_G_INHB);
}
void LIF::ioParam_noiseAmpE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseAmpE", &lParams.noiseAmpE, 0.0f);
}
void LIF::ioParam_noiseAmpI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseAmpI", &lParams.noiseAmpI, 0.0f);
}
void LIF::ioParam_noiseAmpIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseAmpIB", &lParams.noiseAmpIB, 0.0f);
}

void LIF::ioParam_noiseFreqE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseFreqE", &lParams.noiseFreqE, 250.0f);
   if (ioFlag == PARAMS_IO_READ) {
      float dt_sec = 0.001f * (float)parent->getDeltaTime(); // seconds
      if (dt_sec * lParams.noiseFreqE > 1.0f) {
         lParams.noiseFreqE = 1.0f / dt_sec;
      }
   }
}

void LIF::ioParam_noiseFreqI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseFreqI", &lParams.noiseFreqI, 250.0f);
   if (ioFlag == PARAMS_IO_READ) {
      float dt_sec = 0.001f * (float)parent->getDeltaTime(); // seconds
      if (dt_sec * lParams.noiseFreqI > 1.0f) {
         lParams.noiseFreqI = 1.0f / dt_sec;
      }
   }
}

void LIF::ioParam_noiseFreqIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "noiseFreqIB", &lParams.noiseFreqIB, 250.0f);
   if (ioFlag == PARAMS_IO_READ) {
      float dt_sec = 0.001f * (float)parent->getDeltaTime(); // seconds
      if (dt_sec * lParams.noiseFreqIB > 1.0f) {
         lParams.noiseFreqIB = 1.0f / dt_sec;
      }
   }
}

void LIF::ioParam_method(enum ParamsIOFlag ioFlag) {
   // Read the integration method: one of 'arma' (preferred), 'beginning' (deprecated), or
   // 'original' (deprecated).
   const char *default_method = "arma";
   parameters()->ioParamString(
         ioFlag, name, "method", &methodString, default_method, true /*warnIfAbsent*/);
   if (ioFlag != PARAMS_IO_READ) {
      return;
   }

   assert(methodString);
   if (methodString[0] == '\0') {
      free(methodString);
      methodString = strdup(default_method);
      if (methodString == NULL) {
         Fatal().printf(
               "%s: unable to set method string: %s\n", getDescription_c(), strerror(errno));
      }
   }
   method = methodString ? methodString[0]
                         : 'a'; // Default is ARMA; 'beginning' and 'original' are deprecated.
   if (method != 'o' && method != 'b' && method != 'a') {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog().printf(
               "LIF::setLIFParams error.  Layer \"%s\" has method \"%s\".  Allowable values are "
               "\"arma\", \"beginning\" and \"original\".",
               name,
               methodString);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (method != 'a') {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         WarnLog().printf(
               "LIF layer \"%s\" integration method \"%s\" is deprecated.  Method \"arma\" is "
               "preferred.\n",
               name,
               methodString);
      }
   }
}

LayerInputBuffer *LIF::createLayerInput() { return new LIFLayerInputBuffer(name, parent); }

int LIF::setActivity() {
   float *activity = mActivity->getActivity();
   memset(activity, 0, sizeof(float) * getNumExtendedAllBatches());
   return 0;
}

Response::Status
LIF::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

Response::Status LIF::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   // // a random state variable is needed for every neuron/clthread
   randState = new Random(getLayerLoc(), false /*isExtended*/);
   if (randState == nullptr) {
      Fatal().printf(
            "LIF::initialize:  %s unable to create object of Random class.\n", getDescription_c());
   }

   int numNeurons = getNumNeuronsAllBatches();
   assert(Vth); // Allocated when HyPerLayer::allocateDataStructures() called allocateBuffers().
   for (size_t k = 0; k < numNeurons; k++) {
      Vth[k] = lParams.VthRest; // lParams.VthRest is set in setLIFParams
   }
   auto *lifLayerInput = getComponentByType<LIFLayerInputBuffer>();
   FatalIf(lifLayerInput == nullptr, "%s could not find a LIFLayerInput component.\n");
   pvAssert(lifLayerInput->getDataStructuresAllocatedFlag());
   lParams.tauE  = lifLayerInput->getChannelTimeConstant(CHANNEL_EXC);
   lParams.tauI  = lifLayerInput->getChannelTimeConstant(CHANNEL_INH);
   lParams.tauIB = lifLayerInput->getChannelTimeConstant(CHANNEL_INHB);
   return Response::SUCCESS;
}

void LIF::allocateBuffers() {
   allocateConductances(mLayerInput->getNumChannels());
   Vth = (float *)calloc((size_t)getNumNeuronsAllBatches(), sizeof(float));
   if (Vth == NULL) {
      Fatal().printf(
            "%s: rank %d process unable to allocate memory for Vth: %s\n",
            getDescription_c(),
            parent->getCommunicator()->globalCommRank(),
            strerror(errno));
   }
   HyPerLayer::allocateBuffers();
}

void LIF::allocateConductances(int num_channels) {
   pvAssert(num_channels >= 3); // Need exc, inh, and inhb at a minimum.
   const int numNeurons = getNumNeuronsAllBatches();
   G_E = (float *)calloc((size_t)(getNumNeuronsAllBatches() * num_channels), sizeof(float));
   if (G_E == NULL) {
      Fatal().printf(
            "%s: rank %d process unable to allocate memory for %d conductances: %s\n",
            getDescription_c(),
            parent->getCommunicator()->globalCommRank(),
            num_channels,
            strerror(errno));
   }

   G_I  = G_E + 1 * numNeurons;
   G_IB = G_E + 2 * numNeurons;
}

Response::Status LIF::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mInitializeFromCheckpointFlag) {
      auto status = HyPerLayer::readStateFromCheckpoint(checkpointer);
      if (!Response::completed(status)) {
         return status;
      }
      readVthFromCheckpoint(checkpointer);
      readG_EFromCheckpoint(checkpointer);
      readG_IFromCheckpoint(checkpointer);
      readG_IBFromCheckpoint(checkpointer);
      readRandStateFromCheckpoint(checkpointer);
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void LIF::readVthFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), "Vth", false /*not constant*/);
}

void LIF::readG_EFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), "G_E", false /*not constant*/);
}

void LIF::readG_IFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), "G_I", false /*not constant*/);
}

void LIF::readG_IBFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), "G_IB", false /*not constant*/);
}

void LIF::readRandStateFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), "rand_state", false /*not constant*/);
}

Response::Status
LIF::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = HyPerLayer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   checkpointPvpActivityFloat(checkpointer, "Vth", Vth, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "G_E", G_E, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "G_I", G_I, false /*not extended*/);
   checkpointPvpActivityFloat(checkpointer, "G_IB", G_IB, false /*not extended*/);
   checkpointRandState(checkpointer, "rand_state", randState, false /*not extended*/);
   return Response::SUCCESS;
}

Response::Status LIF::updateState(double time, double dt) {
   update_timer->start();

   const int nx     = getLayerLoc()->nx;
   const int ny     = getLayerLoc()->ny;
   const int nf     = getLayerLoc()->nf;
   const int nbatch = getLayerLoc()->nbatch;

   float const *GSynHead = mLayerInput->getBufferData();
   float *activity       = mActivity->getActivity();

   switch (method) {
      case 'a':
         LIF_update_state_arma(
               nbatch,
               getNumNeurons(),
               time,
               dt,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &lParams,
               randState->getRNG(0),
               getV(),
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               activity);
         break;
      case 'b':
         LIF_update_state_beginning(
               nbatch,
               getNumNeurons(),
               time,
               dt,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &lParams,
               randState->getRNG(0),
               getV(),
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               activity);
         break;
      case 'o':
         LIF_update_state_original(
               nbatch,
               getNumNeurons(),
               time,
               dt,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &lParams,
               randState->getRNG(0),
               getV(),
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               activity);
         break;
      default: assert(0); break;
   }
   update_timer->stop();
   return Response::SUCCESS;
}

int LIF::findPostSynaptic(
      int dim,
      int maxSize,
      int col,
      // input: which layer, which neuron
      HyPerLayer *lSource,
      float pos[],

      // output: how many of our neurons are connected.
      // an array with their indices.
      // an array with their feature vectors.
      int *nNeurons,
      int nConnectedNeurons[],
      float *vPos) {
   return 0;
}

} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of LIF kernels
//

inline float LIF_Vmem_derivative(
      const float Vmem,
      const float G_E,
      const float G_I,
      const float G_IB,
      const float V_E,
      const float V_I,
      const float V_IB,
      const float Vrest,
      const float tau) {
   float totalconductance = 1.0f + G_E + G_I + G_IB;
   float Vmeminf          = (Vrest + V_E * G_E + V_I * G_I + V_IB * G_IB) / totalconductance;
   return totalconductance * (Vmeminf - Vmem) / tau;
}
//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
// LIF_update_state_original uses an Euler scheme for V where the conductances over the entire
// timestep are taken to be the values calculated at the end of the timestep
// LIF_update_state_beginning uses a Heun scheme for V, using values of the conductances at both the
// beginning and end of the timestep.  Spikes in the input are applied at the beginning of the
// timestep.
// LIF_update_state_arma uses an auto-regressive moving average filter for V, applying the GSyn at
// the start of the timestep and assuming that tau_inf and V_inf vary linearly over the timestep.
// See van Hateren, Journal of Vision (2005), p. 331.
//
void LIF_update_state_original(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity) {
   int k;

   const float exp_tauE   = expf(-dt / params->tauE);
   const float exp_tauI   = expf(-dt / params->tauI);
   const float exp_tauIB  = expf(-dt / params->tauIB);
   const float exp_tauVth = expf(-dt / params->tauVth);

   const float dt_sec = 0.001f * dt; // convert to seconds

   for (k = 0; k < nx * ny * nf * nbatch; k++) {
      int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //

      // local param variables
      float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

      const float GMAX = 10.0f;

      // local variables
      float l_activ;

      taus_uint4 l_rnd = rnd[k];

      float l_V   = V[k];
      float l_Vth = Vth[k];

      float l_G_E  = G_E[k];
      float l_G_I  = G_I[k];
      float l_G_IB = G_IB[k];

      float const *GSynExc  = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
      float const *GSynInh  = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
      float const *GSynInhB = &GSynHead[CHANNEL_INHB * nbatch * numNeurons];
      float l_GSynExc       = GSynExc[k];
      float l_GSynInh       = GSynInh[k];
      float l_GSynInhB      = GSynInhB[k];

      // temporary arrays
      float tauInf, VmemInf;

      //
      // start of LIF2_update_exact_linear
      //

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

      // add noise
      //

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqE) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynExc = l_GSynExc + params->noiseAmpE * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqI) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynInh = l_GSynInh + params->noiseAmpI * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqIB) {
         l_rnd      = cl_random_get(l_rnd);
         l_GSynInhB = l_GSynInhB + params->noiseAmpIB * cl_random_prob(l_rnd);
      }

      l_G_E  = l_GSynExc + l_G_E * exp_tauE;
      l_G_I  = l_GSynInh + l_G_I * exp_tauI;
      l_G_IB = l_GSynInhB + l_G_IB * exp_tauIB;

      l_G_E  = (l_G_E > GMAX) ? GMAX : l_G_E;
      l_G_I  = (l_G_I > GMAX) ? GMAX : l_G_I;
      l_G_IB = (l_G_IB > GMAX) ? GMAX : l_G_IB;

      tauInf  = (dt / tau) * (1.0f + l_G_E + l_G_I + l_G_IB);
      VmemInf = (Vrest + l_G_E * Vexc + l_G_I * Vinh + l_G_IB * VinhB)
                / (1.0f + l_G_E + l_G_I + l_G_IB);

      l_V = VmemInf + (l_V - VmemInf) * expf(-tauInf);

      //
      // start of LIF2_update_finish
      //

      l_Vth = VthRest + (l_Vth - VthRest) * exp_tauVth;

      //
      // start of update_f
      //

      bool fired_flag = (l_V > l_Vth);

      l_activ = fired_flag ? 1.0f : 0.0f;
      l_V     = fired_flag ? Vrest : l_V;
      l_Vth   = fired_flag ? l_Vth + deltaVth : l_Vth;
      l_G_IB  = fired_flag ? l_G_IB + deltaGIB : l_G_IB;

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

void LIF_update_state_beginning(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity) {
   int k;

   const float exp_tauE   = expf(-dt / params->tauE);
   const float exp_tauI   = expf(-dt / params->tauI);
   const float exp_tauIB  = expf(-dt / params->tauIB);
   const float exp_tauVth = expf(-dt / params->tauVth);

   const float dt_sec = 0.001f * dt; // convert to seconds

   for (k = 0; k < nx * ny * nf * nbatch; k++) {

      int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //

      // local param variables
      float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

      const float GMAX = 10.0f;

      // local variables
      float l_activ;

      taus_uint4 l_rnd = rnd[k];

      float l_V   = V[k];
      float l_Vth = Vth[k];

      // The correction factors to the conductances are so that if l_GSyn_* is the same every
      // timestep,
      // then the asymptotic value of l_G_* will be l_GSyn_*
      float l_G_E  = G_E[k];
      float l_G_I  = G_I[k];
      float l_G_IB = G_IB[k];

      float const *GSynExc  = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
      float const *GSynInh  = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
      float const *GSynInhB = &GSynHead[CHANNEL_INHB * nbatch * numNeurons];
      float l_GSynExc       = GSynExc[k];
      float l_GSynInh       = GSynInh[k];
      float l_GSynInhB      = GSynInhB[k];

      //
      // start of LIF2_update_exact_linear
      //

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

      // add noise
      //

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqE) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynExc = l_GSynExc + params->noiseAmpE * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqI) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynInh = l_GSynInh + params->noiseAmpI * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqIB) {
         l_rnd      = cl_random_get(l_rnd);
         l_GSynInhB = l_GSynInhB + params->noiseAmpIB * cl_random_prob(l_rnd);
      }

      // The portion of code below uses the newer method of calculating l_V.
      float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
      float dV1, dV2, dV;

      G_E_initial  = l_G_E + l_GSynExc;
      G_I_initial  = l_G_I + l_GSynInh;
      G_IB_initial = l_G_IB + l_GSynInhB;

      G_E_initial  = (G_E_initial > GMAX) ? GMAX : G_E_initial;
      G_I_initial  = (G_I_initial > GMAX) ? GMAX : G_I_initial;
      G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

      G_E_final  = G_E_initial * exp_tauE;
      G_I_final  = G_I_initial * exp_tauI;
      G_IB_final = G_IB_initial * exp_tauIB;

      dV1 = LIF_Vmem_derivative(
            l_V, G_E_initial, G_I_initial, G_IB_initial, Vexc, Vinh, VinhB, Vrest, tau);
      dV2 = LIF_Vmem_derivative(
            l_V + dt * dV1, G_E_final, G_I_final, G_IB_final, Vexc, Vinh, VinhB, Vrest, tau);
      dV  = (dV1 + dV2) * 0.5f;
      l_V = l_V + dt * dV;

      l_G_E  = G_E_final;
      l_G_I  = G_I_final;
      l_G_IB = G_IB_final;

      l_Vth = VthRest + (l_Vth - VthRest) * exp_tauVth;
      // End of code unique to newer method.

      //
      // start of update_f
      //

      bool fired_flag = (l_V > l_Vth);

      l_activ = fired_flag ? 1.0f : 0.0f;
      l_V     = fired_flag ? Vrest : l_V;
      l_Vth   = fired_flag ? l_Vth + deltaVth : l_Vth;
      l_G_IB  = fired_flag ? l_G_IB + deltaGIB : l_G_IB;

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

void LIF_update_state_arma(
      const int nbatch,
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

      LIF_params *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *activity) {
   int k;

   const float exp_tauE   = expf(-dt / params->tauE);
   const float exp_tauI   = expf(-dt / params->tauI);
   const float exp_tauIB  = expf(-dt / params->tauIB);
   const float exp_tauVth = expf(-dt / params->tauVth);

   const float dt_sec = 0.001f * dt; // convert to seconds

   for (k = 0; k < nx * ny * nf * nbatch; k++) {
      int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //

      // local param variables
      float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

      const float GMAX = 10.0;

      // local variables
      float l_activ;

      taus_uint4 l_rnd = rnd[k];

      float l_V   = V[k];
      float l_Vth = Vth[k];

      // The correction factors to the conductances are so that if l_GSyn_* is the same every
      // timestep,
      // then the asymptotic value of l_G_* will be l_GSyn_*
      float l_G_E  = G_E[k];
      float l_G_I  = G_I[k];
      float l_G_IB = G_IB[k];

      float const *GSynExc  = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
      float const *GSynInh  = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
      float const *GSynInhB = &GSynHead[CHANNEL_INHB * nbatch * numNeurons];
      float l_GSynExc       = GSynExc[k];
      float l_GSynInh       = GSynInh[k];
      float l_GSynInhB      = GSynInhB[k];

      //
      // start of LIF2_update_exact_linear
      //

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

      // add noise
      //

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqE) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynExc = l_GSynExc + params->noiseAmpE * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqI) {
         l_rnd     = cl_random_get(l_rnd);
         l_GSynInh = l_GSynInh + params->noiseAmpI * cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec * params->noiseFreqIB) {
         l_rnd      = cl_random_get(l_rnd);
         l_GSynInhB = l_GSynInhB + params->noiseAmpIB * cl_random_prob(l_rnd);
      }

      // The portion of code below uses the newer method of calculating l_V.
      float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
      float tau_inf_initial, tau_inf_final, V_inf_initial, V_inf_final;

      G_E_initial     = l_G_E + l_GSynExc;
      G_I_initial     = l_G_I + l_GSynInh;
      G_IB_initial    = l_G_IB + l_GSynInhB;
      tau_inf_initial = tau / (1 + G_E_initial + G_I_initial + G_IB_initial);
      V_inf_initial   = (Vrest + Vexc * G_E_initial + Vinh * G_I_initial + VinhB * G_IB_initial)
                      / (1 + G_E_initial + G_I_initial + G_IB_initial);

      G_E_initial  = (G_E_initial > GMAX) ? GMAX : G_E_initial;
      G_I_initial  = (G_I_initial > GMAX) ? GMAX : G_I_initial;
      G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

      G_E_final  = G_E_initial * exp_tauE;
      G_I_final  = G_I_initial * exp_tauI;
      G_IB_final = G_IB_initial * exp_tauIB;

      tau_inf_final = tau / (1 + G_E_final + G_I_final + G_IB_initial);
      V_inf_final   = (Vrest + Vexc * G_E_final + Vinh * G_I_final + VinhB * G_IB_final)
                    / (1 + G_E_final + G_I_final + G_IB_final);

      float tau_slope = (tau_inf_final - tau_inf_initial) / dt;
      float f1        = tau_slope == 0.0f ? expf(-dt / tau_inf_initial)
                                   : powf(tau_inf_final / tau_inf_initial, -1 / tau_slope);
      float f2 = tau_slope == -1.0f
                       ? tau_inf_initial / dt * logf(tau_inf_final / tau_inf_initial + 1.0f)
                       : (1 - tau_inf_initial / dt * (1 - f1)) / (1 + tau_slope);
      float f3 = 1.0f - f1 - f2;
      l_V      = f1 * l_V + f2 * V_inf_initial + f3 * V_inf_final;

      l_G_E  = G_E_final;
      l_G_I  = G_I_final;
      l_G_IB = G_IB_final;

      l_Vth = VthRest + (l_Vth - VthRest) * exp_tauVth;
      // End of code unique to newer method.

      //
      // start of update_f
      //

      bool fired_flag = (l_V > l_Vth);

      l_activ = fired_flag ? 1.0f : 0.0f;
      l_V     = fired_flag ? Vrest : l_V;
      l_Vth   = fired_flag ? l_Vth + deltaVth : l_Vth;
      l_G_IB  = fired_flag ? l_G_IB + deltaGIB : l_G_IB;

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
   }
}
