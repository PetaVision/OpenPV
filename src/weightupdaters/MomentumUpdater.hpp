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
    * dW = (1-tauFactor) * dW_Hebb + tauFactor * dWprev - weightL2Decay * W.
    *
    * where dW_Hebb = dWMax * pre * post and tauFactor = exp(-1/timeConstantTau).
    * The interpretation is that for the impulse from a single pre*post event,
    * the weight approaches the value dW_Hebb as t->infinity, and that timeConstantTau
    * indicates the rate of decay of W - dW_Hebb, measured in weightUpdatePeriods
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
    * weightL2Decay: The L2-driven decay rate on the weights, applied after the momentum updates.
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

   void applyMomentum(int arborId, float dwFactor, float wFactor);

   void openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   virtual void outputMomentum(double timestamp);

  protected:
   enum Method { UNDEFINED_METHOD, VISCOSITY, SIMPLE };

   char *mMomentumMethod    = nullptr;
   Method mMethod           = UNDEFINED_METHOD;
   float mTimeConstantTau   = mDefaultTimeConstantTauViscosity;
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
