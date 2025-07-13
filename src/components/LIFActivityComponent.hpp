/*
 * LIFActivityComponent.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#ifndef LIFACTIVITYCOMPONENT_HPP_
#define LIFACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"

#include "columns/Random.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/RestrictedBuffer.hpp"
#include "include/default_params.h"
#include "utils/cl_random.h"

namespace PV {

struct LIFParams {

   float Vrest = (float)V_REST;
   float Vexc  = (float)V_EXC;
   float Vinh  = (float)V_INH;
   float VinhB = (float)V_INHB;

   float VthRest  = (float)VTH_REST;
   float tau      = (float)TAU_VMEM;
   float tauVth   = (float)TAU_VTH;
   float deltaVth = (float)DELTA_VTH;
   float deltaGIB = (float)DELTA_G_INHB;

   float noiseAmpE   = 0.0f;
   float noiseAmpI   = 0.0f;
   float noiseAmpIB  = 0.0f;
   float noiseFreqE  = 250.0f;
   float noiseFreqI  = 250.0f;
   float noiseFreqIB = 250.0f;

   float tauE  = (float)TAU_EXC;
   float tauI  = (float)TAU_INH;
   float tauIB = (float)TAU_INHB;
};

/**
 * The base class for layer buffers such as GSyn, membrane potential, activity, etc.
 */
class LIFActivityComponent : public ActivityComponent {
  protected:
   virtual void ioParam_Vrest(ParamsIOSwitch ioSwitch);
   virtual void ioParam_Vexc(ParamsIOSwitch ioSwitch);
   virtual void ioParam_Vinh(ParamsIOSwitch ioSwitch);
   virtual void ioParam_VinhB(ParamsIOSwitch ioSwitch);
   virtual void ioParam_tau(ParamsIOSwitch ioSwitch);
   virtual void ioParam_VthRest(ParamsIOSwitch ioSwitch);
   virtual void ioParam_tauVth(ParamsIOSwitch ioSwitch);
   virtual void ioParam_deltaVth(ParamsIOSwitch ioSwitch);
   virtual void ioParam_deltaGIB(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseAmpE(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseAmpI(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseAmpIB(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseFreqE(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseFreqI(ParamsIOSwitch ioSwitch);
   virtual void ioParam_noiseFreqIB(ParamsIOSwitch ioSwitch);

   /** @brief tauE: the time constant for the excitatory channel. */
   virtual void ioParam_tauE(ParamsIOSwitch ioSwitch);

   /** @brief tauI: the time constant for the inhibitory channel. */
   virtual void ioParam_tauI(ParamsIOSwitch ioSwitch);

   /** @brief tauIB: the time constant for the after-hyperpolarization. */
   virtual void ioParam_tauIB(ParamsIOSwitch ioSwitch);
   virtual void ioParam_method(ParamsIOSwitch ioSwitch);

  public:
   LIFActivityComponent(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~LIFActivityComponent();

  protected:
   LIFActivityComponent() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual void fillComponentTable() override;

   virtual RestrictedBuffer *createRestrictedBuffer(char const *label);

   virtual InternalStateBuffer *createInternalState();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   /**
    * Called by ioParam_method. Gives a fatal error if the method string is not one of
    * "arma", "beginning", or "original", and gives a warning if it is "beginning", or "original".
    * (Technically, it only checks the first character against 'a', 'b', or 'o'.)
    */
   void checkMethodString();

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   // TODO: Eliminate code duplication with Retina - Make RandState a component
   void registerRandState(Checkpointer *checkpointer);

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

   void updateActivityArma(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const dt,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *A);

   void updateActivityBeginning(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const dt,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *A);

   void updateActivityOriginal(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const dt,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *A);

  protected:
   LIFParams mLIFParams; // used in update state
   std::string mMethodString; // 'arma', 'before', or 'original'
   char mMethod = 'a'; // 'a', 'b', or 'o', the first character of methodString

   RestrictedBuffer *mConductanceE     = nullptr;
   RestrictedBuffer *mConductanceI     = nullptr;
   RestrictedBuffer *mConductanceIB    = nullptr;
   InternalStateBuffer *mInternalState = nullptr;
   LayerInputBuffer *mLayerInput       = nullptr;
   Random *mRandState                  = nullptr;
   RestrictedBuffer *mVth              = nullptr;
};

} // namespace PV

#endif // LIFACTIVITYCOMPONENT_HPP_
