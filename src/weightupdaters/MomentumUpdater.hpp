/*
 * MomentumUpdater.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMUPDATER_HPP_
#define MOMENTUMUPDATER_HPP_

#include "weightupdaters/HebbianUpdater.hpp"

#include "io/WeightsFile.hpp"
#include "structures/Weights.hpp"
#include <memory>

namespace PV {

/**
 * MomentumUpdater is a weight update class that adds momentum and elastic-net decay to
 * the Hebbian update rule. The parameters for specifying the update rule are
 * momentumMethod (either "viscosity" or "simple"), timeConstantTau (to specify momentum),
 * weightL1Decay (for the L1-like decay term), and weightL2Decay (for the L2-like decay term).
 *
 * The weight update quantity dW is computed as follows:
 * For each weight W, calculate the Hebbian update quantity
 *    dW_Hebb = dWMax * pre * post / numActivations
 * where pre and post are the presynaptic and postsynaptic activities, dWMax is the parameter by
 * that name, and numActivations is the number of kernel activations if using shared weights, and
 * one if using local-patch weights.
 *
 * The momentum term of dW is then
 *    dW_momentum = (1-tauFactor) * dW_Hebb + tauFactor * dW_prev,
 * where dW_prev is the value of dW on the previous update period.
 * For the meaning of tauFactor, see the description of the timeConstantTau parameter.
 *
 * The L1-decay L2-decay terms are given in the description of weightL1Decay and weightL2Decay:
 *    dW_L1Decay = -sgn(W) * min(|W|, weightL1Decay).
 *    dW_L2Decay = -weightL1Decay * W.
 *
 * The update quantity dW is then dW_momentum + dW_L1Decay + dW_L2Decay.
 * Finally, W is updated to W + dW, and dW is copied to dW_prev.
 */
class MomentumUpdater : public HebbianUpdater {
  protected:
   /**
    * List of parameters needed from the MomentumUpdater class
    * @name MomentumUpdater Parameters
    * @{
    */

   /**
    * @brief momentumMethod: Controls the interpretation of the timeConstantTau and momentumDelay
    * parameters.
    */
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief timeConstantTau: controls the amount of momentum in weight updates.
    * @details The contribution to dW from the Hebbian update rule with momentum is:
    *
    *   (1-tauFactor) * dW_Hebb + tauFactor * dW_prev
    *
    * where dW_Hebb = dWMax * pre * post (normalized by the number of kernel updates) and
    * tauFactor = exp(-1/timeConstantTau) for momentumMethod "viscosity", and
    * tauFactor = timeConstantTau for momentumMethod "simple".
    *
    * The interpretation for "viscosity" is that for the impulse from a single pre*post event,
    * the weight approaches the value dW_Hebb as t->infinity, and that timeConstantTau indicates
    * the rate of decay of W - dW_Hebb, measured in weightUpdatePeriods
    *
    * For momentumMethod = "simple", the update rule is the same as for "viscosity",
    * except that tauFactor = timeConstantTau.
    *
    * For momentumMethod = "simple", 0 <= timeConstantTau < 1 is required.
    * For momentumMethod = "viscosity", timeConstantTau >= 0 is required.
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);

   /**
    * @brief momentumTau is obsolete. Use timeConstantTau instead.
    * If a momentum connection sets the momentumTau parameter, it is a fatal error
    * and the error message advises to use momentumTau instead.
    */
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);

   /**
    * momentumDecay is a deprecated synonym for weightL2Decay: Use weightL2Decay instead.
    */
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);

   /**
    * weightL1Decay: The L1-driven decay rate on the weights, applied after the momentum updates.
    * @details The contribution to dW from L1-driven decay is:
    *
    *   -sgn(W) * min(|W|, weightL1Decay).
    *
    * The default value is zero (no L1-decay). It is an error for weightL1Decay to be negative.
    */
   virtual void ioParam_weightL1Decay(enum ParamsIOFlag ioFlag);

   /**
    * weightL2Decay: The L2-driven decay rate on the weights, applied after the momentum updates.
    * @details The contribution to dW from L1-driven decay is:
    *
    *   -weightL1Decay * W.
    *
    * The default value is zero (no L2-decay). It is an error for weightL2Decay to be negative.
    */
   virtual void ioParam_weightL2Decay(enum ParamsIOFlag ioFlag);

   /**
    * initPrev_dWFile: The .pvp file to read initial values of prev_dW used when applying momentum.
    * NULL or the empty string initialzies prev_dW to all zeroes.
    */
   virtual void ioParam_initPrev_dWFile(enum ParamsIOFlag ioFlag);

   /**
    * prev_dWFrameNumber: The frame number (zero-indexed) to use when reading initial prev_dW.
    * default is zero.
    */
   virtual void ioParam_prev_dWFrameNumber(enum ParamsIOFlag ioFlag);

   /** @} */ // end of MomentumUpdater parameters

  public:
   // default values for timeConstantTau
   static constexpr float mDefaultTimeConstantTauSimple    = 0.25f;
   static constexpr float mDefaultTimeConstantTauViscosity = 100.0f;

   MomentumUpdater(char const *name, PVParams *params, Communicator const *comm);

   virtual ~MomentumUpdater() {}

   char const *getMomentumMethod() { return mMomentumMethod; }
   float getTimeConstantTau() const { return mTimeConstantTau; }

   Weights const *getPrevDeltaWeights() const { return mPrevDeltaWeights; }

  protected:
   MomentumUpdater() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void checkTimeConstantTau();

   virtual void initMessageActionMap() override;

   Response::Status respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual int updateWeights(int arborId) override;

   void applyMomentum(int arborId);

   void applyMomentum(int arborId, float dwFactor);

   void openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   virtual void outputMomentum(double timestamp);

  protected:
   enum Method { UNDEFINED_METHOD, VISCOSITY, SIMPLE };

   char *mMomentumMethod    = nullptr;
   Method mMethod           = UNDEFINED_METHOD;
   float mTimeConstantTau   = mDefaultTimeConstantTauViscosity;
   float mWeightL1Decay     = 0.0f;
   float mWeightL2Decay     = 0.0f;
   char *mInitPrev_dWFile   = nullptr;
   int  mPrev_dWFrameNumber = 0;

   Weights *mPrevDeltaWeights       = nullptr;

   // Copied from WeightsPair for use by outputMomentum
   double mWriteStep = 0.0;
   double mWriteTime = 0.0;
   bool mWriteCompressedWeights = false;

   std::shared_ptr<WeightsFile> mWeightsFile;
};

} // namespace PV

#endif // MOMENTUMUPDATER_HPP_
