/*
 * LIFActivityComponent.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "LIFActivityComponent.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "components/InternalStateBuffer.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

LIFActivityComponent::LIFActivityComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

LIFActivityComponent::~LIFActivityComponent() { delete mRandState; }

void LIFActivityComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityComponent::initialize(name, params, comm);
}

void LIFActivityComponent::setObjectType() { mObjectType = "LIFActivityComponent"; }

void LIFActivityComponent::fillComponentTable() {
   ActivityComponent::fillComponentTable(); // creates A and V buffers
   mInternalState = createInternalState();
   if (mInternalState) {
      addUniqueComponent(mInternalState);
   }
   mConductanceE = createRestrictedBuffer("G_E");
   if (mConductanceE) {
      addObserver(std::string(mConductanceE->getName()), mConductanceE);
   }
   mConductanceI = createRestrictedBuffer("G_I");
   if (mConductanceI) {
      addObserver(std::string(mConductanceI->getName()), mConductanceI);
   }
   mConductanceIB = createRestrictedBuffer("G_IB");
   if (mConductanceIB) {
      addObserver(std::string(mConductanceIB->getName()), mConductanceIB);
   }
   mVth = createRestrictedBuffer("Vth");
   if (mVth) {
      addObserver(std::string(mVth->getName()), mVth);
   }
}

int LIFActivityComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ActivityComponent::ioParamsFillGroup(ioFlag);

   ioParam_Vrest(ioFlag);
   ioParam_Vexc(ioFlag);
   ioParam_Vinh(ioFlag);
   ioParam_VinhB(ioFlag);
   ioParam_VthRest(ioFlag);
   ioParam_tau(ioFlag);
   ioParam_tauVth(ioFlag);
   ioParam_deltaVth(ioFlag);
   ioParam_deltaGIB(ioFlag);
   ioParam_noiseAmpE(ioFlag);
   ioParam_noiseAmpI(ioFlag);
   ioParam_noiseAmpIB(ioFlag);
   ioParam_noiseFreqE(ioFlag);
   ioParam_noiseFreqI(ioFlag);
   ioParam_noiseFreqIB(ioFlag);
   ioParam_tauE(ioFlag);
   ioParam_tauI(ioFlag);
   ioParam_tauIB(ioFlag);
   ioParam_method(ioFlag);
   return PV_SUCCESS;
}

void LIFActivityComponent::ioParam_Vrest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vrest", &mLIFParams.Vrest, mLIFParams.Vrest);
}

void LIFActivityComponent::ioParam_Vexc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vexc", &mLIFParams.Vexc, mLIFParams.Vexc);
}

void LIFActivityComponent::ioParam_Vinh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "Vinh", &mLIFParams.Vinh, mLIFParams.Vinh);
}

void LIFActivityComponent::ioParam_VinhB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VinhB", &mLIFParams.VinhB, mLIFParams.VinhB);
}

void LIFActivityComponent::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "VthRest", &mLIFParams.VthRest, mLIFParams.VthRest);
}

void LIFActivityComponent::ioParam_tau(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tau", &mLIFParams.tau, mLIFParams.tau);
}

void LIFActivityComponent::ioParam_tauVth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauVth", &mLIFParams.tauVth, mLIFParams.tauVth);
}

void LIFActivityComponent::ioParam_deltaVth(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "deltaVth", &mLIFParams.deltaVth, mLIFParams.deltaVth);
}

void LIFActivityComponent::ioParam_deltaGIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "deltaGIB", &mLIFParams.deltaGIB, mLIFParams.deltaGIB);
}

void LIFActivityComponent::ioParam_noiseAmpE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseAmpE", &mLIFParams.noiseAmpE, mLIFParams.noiseAmpE);
}

void LIFActivityComponent::ioParam_noiseAmpI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseAmpI", &mLIFParams.noiseAmpI, mLIFParams.noiseAmpI);
}

void LIFActivityComponent::ioParam_noiseAmpIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseAmpIB", &mLIFParams.noiseAmpIB, mLIFParams.noiseAmpIB);
}

void LIFActivityComponent::ioParam_noiseFreqE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseFreqE", &mLIFParams.noiseFreqE, mLIFParams.noiseFreqE);
   // Truncation to 1/(0.001*dt) has been moved to initializeState() method.
}

void LIFActivityComponent::ioParam_noiseFreqI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseFreqI", &mLIFParams.noiseFreqI, mLIFParams.noiseFreqI);
   // Truncation to 1/(0.001*dt) has been moved to initializeState() method.
}

void LIFActivityComponent::ioParam_noiseFreqIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "noiseFreqIB", &mLIFParams.noiseFreqIB, mLIFParams.noiseFreqIB);
   // Truncation to 1/(0.001*dt) has been moved to initializeState() method.
}

void LIFActivityComponent::ioParam_tauE(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauE", &mLIFParams.tauE, mLIFParams.tauE);
}

void LIFActivityComponent::ioParam_tauI(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauI", &mLIFParams.tauI, mLIFParams.tauI);
}

void LIFActivityComponent::ioParam_tauIB(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauIB", &mLIFParams.tauIB, mLIFParams.tauIB);
}

void LIFActivityComponent::ioParam_method(enum ParamsIOFlag ioFlag) {
   // Read the integration method: one of 'arma' (preferred), 'beginning' (deprecated), or
   // 'original' (deprecated).
   char const *defaultMethod = "arma";
   parameters()->ioParamString(
         ioFlag, name, "method", &mMethodString, defaultMethod, true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      pvAssert(mMethodString);
      if (mMethodString[0] == '\0') {
         free(mMethodString);
         mMethodString = strdup(defaultMethod);
         if (mMethodString == nullptr) {
            Fatal().printf(
                  "%s: unable to set method string: %s\n", getDescription_c(), strerror(errno));
         }
      }
      checkMethodString();
   }
}

void LIFActivityComponent::checkMethodString() {
   pvAssert(mMethodString);
   mMethod = mMethodString[0];
   if (mMethod != 'o' && mMethod != 'b' && mMethod != 'a') {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "LIFActivityComponent::ioParam_method error.  Layer \"%s\" has method \"%s\".  "
               "Allowable values are \"arma\", \"beginning\" and \"original\".\n",
               name,
               mMethodString);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   if (mMethod != 'a') {
      if (mCommunicator->globalCommRank() == 0) {
         WarnLog().printf(
               "LIF layer \"%s\" integration method \"%s\" is deprecated.  Method \"arma\" is "
               "preferred.\n",
               name,
               mMethodString);
      }
   }
}

RestrictedBuffer *LIFActivityComponent::createRestrictedBuffer(char const *label) {
   RestrictedBuffer *buffer = new RestrictedBuffer(getName(), parameters(), mCommunicator);
   buffer->setBufferLabel(label);
   return buffer;
}

InternalStateBuffer *LIFActivityComponent::createInternalState() {
   return new InternalStateBuffer(getName(), parameters(), mCommunicator);
}

Response::Status LIFActivityComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mLayerInput = message->mObjectTable->findObject<LayerInputBuffer>(getName());
   FatalIf(
         mLayerInput == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);
   mLayerInput->requireChannel(CHANNEL_INHB);

   return Response::SUCCESS;
}

Response::Status LIFActivityComponent::allocateDataStructures() {
   auto status = ActivityComponent::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   ComponentBuffer::checkDimensionsEqual(mLayerInput, mConductanceE);
   ComponentBuffer::checkDimensionsEqual(mLayerInput, mConductanceI);
   ComponentBuffer::checkDimensionsEqual(mLayerInput, mConductanceIB);
   ComponentBuffer::checkDimensionsEqual(mLayerInput, mInternalState);
   ComponentBuffer::checkDimensionsEqual(mInternalState, mVth);
   ComponentBuffer::checkDimensionsEqual(mInternalState, mActivity);

   mRandState = new Random(getLayerLoc(), false /*restricted*/);
   return Response::SUCCESS;
}

Response::Status LIFActivityComponent::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ActivityComponent::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   registerRandState(checkpointer);
   return Response::SUCCESS;
}

void LIFActivityComponent::registerRandState(Checkpointer *checkpointer) {
   auto checkpointEntry = std::make_shared<CheckpointEntryRandState>(
         getName(),
         "rand_state",
         checkpointer->getMPIBlock(),
         mRandState->getRNG(0),
         getLayerLoc(),
         false /*restricted buffer*/);
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(checkpointEntry, false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s buffer for checkpointing.\n",
         getDescription_c(),
         "rand_state");
}

Response::Status
LIFActivityComponent::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = ActivityComponent::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   float maxFreq = (float)(1000.0 / message->mDeltaTime); // dt is in ms; frequencies are in Hz
   if (mLIFParams.noiseFreqE > maxFreq) {
      WarnLog().printf(
            "%s noiseFreqE value %f is above the maximum possible value of %f; truncating.\n",
            getDescription_c(),
            (double)mLIFParams.noiseFreqE,
            (double)maxFreq);
      mLIFParams.noiseFreqE = maxFreq;
   }
   if (mLIFParams.noiseFreqI > maxFreq) {
      WarnLog().printf(
            "%s noiseFreqI value %f is above the maximum possible value of %f; truncating.\n",
            getDescription_c(),
            (double)mLIFParams.noiseFreqI,
            (double)maxFreq);
      mLIFParams.noiseFreqI = maxFreq;
   }
   if (mLIFParams.noiseFreqIB > maxFreq) {
      WarnLog().printf(
            "%s noiseFreqIB value %f is above the maximum possible value of %f; truncating.\n",
            getDescription_c(),
            (double)mLIFParams.noiseFreqIB,
            (double)maxFreq);
      mLIFParams.noiseFreqIB = maxFreq;
   }

   float *G_E                      = mConductanceE->getReadWritePointer();
   float *G_I                      = mConductanceI->getReadWritePointer();
   float *G_IB                     = mConductanceIB->getReadWritePointer();
   float *Vth                      = mVth->getReadWritePointer();
   int const numNeuronsAcrossBatch = mInternalState->getBufferSizeAcrossBatch();
   pvAssert(mConductanceE->getBufferSizeAcrossBatch() == numNeuronsAcrossBatch);
   pvAssert(mConductanceI->getBufferSizeAcrossBatch() == numNeuronsAcrossBatch);
   pvAssert(mConductanceIB->getBufferSizeAcrossBatch() == numNeuronsAcrossBatch);
   pvAssert(mVth->getBufferSizeAcrossBatch() == numNeuronsAcrossBatch);
   float const initialVthRest = mLIFParams.VthRest;
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      G_E[k]  = 0.0f;
      G_I[k]  = 0.0f;
      G_IB[k] = 0.0f;
      Vth[k]  = initialVthRest;
   }
   mInternalState->respond(message);
   mActivity->respond(message);
   return Response::SUCCESS;
}

Response::Status LIFActivityComponent::updateActivity(double simTime, double deltaTime) {
   const int nx     = getLayerLoc()->nx;
   const int ny     = getLayerLoc()->ny;
   const int nf     = getLayerLoc()->nf;
   const int nbatch = getLayerLoc()->nbatch;

   float const *GSynHead = mLayerInput->getBufferData();
   float *G_E            = mConductanceE->getReadWritePointer();
   float *G_I            = mConductanceI->getReadWritePointer();
   float *G_IB           = mConductanceIB->getReadWritePointer();
   float *Vth            = mVth->getReadWritePointer();
   float *V              = mInternalState->getReadWritePointer();
   float *A              = mActivity->getReadWritePointer();

   switch (mMethod) {
      case 'a':
         updateActivityArma(
               nbatch,
               nx * ny * nf,
               simTime,
               deltaTime,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &mLIFParams,
               mRandState->getRNG(0),
               V,
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               A);
         break;
      case 'b':
         updateActivityBeginning(
               nbatch,
               nx * ny * nf,
               simTime,
               deltaTime,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &mLIFParams,
               mRandState->getRNG(0),
               V,
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               A);
         break;
      case 'o':
         updateActivityOriginal(
               nbatch,
               nx * ny * nf,
               simTime,
               deltaTime,
               nx,
               ny,
               nf,
               getLayerLoc()->halo.lt,
               getLayerLoc()->halo.rt,
               getLayerLoc()->halo.dn,
               getLayerLoc()->halo.up,
               &mLIFParams,
               mRandState->getRNG(0),
               V,
               Vth,
               G_E,
               G_I,
               G_IB,
               GSynHead,
               A);
         break;
      default: assert(0); break;
   }
   return Response::SUCCESS;
}

// updateActivityOriginal uses an Euler scheme for V where the conductances over the entire
// timestep are taken to be the values calculated at the end of the timestep
// updateActivityBeginning uses a Heun scheme for V, using values of the conductances at both the
// beginning and end of the timestep.  Spikes in the input are applied at the beginning of the
// timestep.
// updateActivityArma uses an auto-regressive moving average filter for V, applying the GSyn at
// the start of the timestep and assuming that tau_inf and V_inf vary linearly over the timestep.
// See van Hateren, Journal of Vision (2005), p. 331.
//
void LIFActivityComponent::updateActivityOriginal(
      const int nbatch,
      const int numNeurons,
      const float simTime,
      const float dt,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      LIFParams *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *A) {
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

      A[kex] = l_activ;

      V[k]   = l_V;
      Vth[k] = l_Vth;

      G_E[k]  = l_G_E;
      G_I[k]  = l_G_I;
      G_IB[k] = l_G_IB;

   } // loop over k
}

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

void LIFActivityComponent::updateActivityBeginning(
      const int nbatch,
      const int numNeurons,
      const float simTime,
      const float dt,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      LIFParams *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *A) {
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

      A[kex] = l_activ;

      V[k]   = l_V;
      Vth[k] = l_Vth;

      G_E[k]  = l_G_E;
      G_I[k]  = l_G_I;
      G_IB[k] = l_G_IB;

   } // loop over k
}

void LIFActivityComponent::updateActivityArma(
      const int nbatch,
      const int numNeurons,
      const float simTime,
      const float dt,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      LIFParams *params,
      taus_uint4 *rnd,
      float *V,
      float *Vth,
      float *G_E,
      float *G_I,
      float *G_IB,
      float const *GSynHead,
      float *A) {
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

      A[kex] = l_activ;

      V[k]   = l_V;
      Vth[k] = l_Vth;

      G_E[k]  = l_G_E;
      G_I[k]  = l_G_I;
      G_IB[k] = l_G_IB;
   }
}

} // namespace PV
