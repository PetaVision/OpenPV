/*
 * MomentumUpdater.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMUPDATER_HPP_
#define MOMENTUMUPDATER_HPP_

#include "weightupdaters/HebbianUpdater.hpp"

#include "checkpointing/CheckpointableFileStream.hpp"

namespace PV {

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
    * @details For momentumMethod = "viscosity", the update rule is
    *
    * dW = (1-tauFactor) * dW_Hebb + tauFactor * dWprev - momentumDecay * W.
    *
    * where dW_Hebb = dWMax * pre * post and tauFactor = exp(-1/timeConstantTau).
    * The interpretation is that for the impulse from a single pre*post event,
    * the weight approaches the value dW_Hebb as t->infinity, and that timeConstantTau
    * indicates the rate of decay of W - dW_Hebb, measured in weightUpdatePeriods
    *
    * For momentumMethod = "simple", the update rule is the same as for "viscosity",
    * except that tauFactor = timeConstantTau.
    *
    * For momentumMethod = "alex", the update rule is
    *
    * dW = ((1-timeConstantTau) * dW_Hebb + timeConstantTau * dWprev) - momentumDecay * dWMax * W,
    *
    * For momentumMethod = "simple" or "alex", 0 <= timeConstantTau < 1 is required.
    * For momentumMethod = "viscosity", timeConstantTau >= 0 is required.
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);

   /**
    * @brief momentumTau: controls the amount of momentum in weight updates.
    * Deprecated in favor of timeConstantTau.
    * @details If timeConstantTau is not in params and momentumTau is,
    * the update rule is as follows.
    *
    * For momentumMethod = "simple", the update rule is
    *
    * dW = dW_Hebb + momentumTau * dWprev - momentumDecay * W.
    *
    * where dW_Hebb = dWMax * pre * post.
    *
    * For momentumMethod = "viscosity", the update rule is the same as for "simple",
    * except that momentumTau is replaced by exp(-1/momentumTau).
    *
    * For momentumMethod = "alex", the update rule is
    *
    * dW = dW_Hebb + momentumTau * dWprev = momentumTau * dWMax * W.
    */
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);

   /**
    * momentumDecay: The decay rate on the weights, applied after the momentum weight updates.
    */
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);

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
   static constexpr float mDefaultTimeConstantTauAlex      = 0.9f;

   MomentumUpdater(char const *name, PVParams *params, Communicator const *comm);

   virtual ~MomentumUpdater() {}

   char const *getMomentumMethod() { return mMomentumMethod; }
   float getTimeConstantTau() const { return mTimeConstantTau; }
   bool isUsingDeprecatedMomentumTau() const { return mUsingDeprecatedMomentumTau; }

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

   void applyMomentum(int arborId, float dwFactor, float wFactor);

   void applyMomentumDeprecated(int arborId);

   void applyMomentumDeprecated(int arborId, float dwFactor, float wFactor);

   void openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   virtual void outputMomentum(double timestamp);

  protected:
   enum Method { UNDEFINED_METHOD, VISCOSITY, SIMPLE, ALEX };

   char *mMomentumMethod    = nullptr;
   Method mMethod           = UNDEFINED_METHOD;
   float mMomentumTau       = 0.25f; // Deprecated in favor of mTimeConstantTau Nov 19, 2018.
   float mTimeConstantTau   = mDefaultTimeConstantTauViscosity;
   float mMomentumDecay     = 0.0f;
   char *mInitPrev_dWFile   = NULL;
   int  mPrev_dWFrameNumber = 0;

   Weights *mPrevDeltaWeights       = nullptr;
   bool mUsingDeprecatedMomentumTau = false;

   // Copied from WeightsPair for use by outputMomentum
   double mWriteStep = 0.0;
   double mWriteTime = 0.0;
   bool mWriteCompressedWeights = false;

   CheckpointableFileStream *mOutputStateStream = nullptr; // weights file written by outputState
};

} // namespace PV

#endif // MOMENTUMUPDATER_HPP_
