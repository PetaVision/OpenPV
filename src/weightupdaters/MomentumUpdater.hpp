/*
 * MomentumUpdater.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMUPDATER_HPP_
#define MOMENTUMUPDATER_HPP_

#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

class MomentumUpdater : public HebbianUpdater {
  protected:
   /**
    * List of parameters needed from the MomentumUpdater class
    * @name MomentumUpdater Parameters
    * @{
    */

   /**
    * @brief momentumMethod: The momentum method to use
    * @details Assuming a = dwMax * pre * post
    * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
    * viscosity: deltaW(t) = (deltaW(t-1) * exp(-1/momentumTau)) + a
    * alex: deltaW(t) = momentumTau * delta(t-1) - momentumDecay * dwMax * w(t) + a
    */
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);

   /** @} */ // end of MomentumUpdater parameters

  public:
   MomentumUpdater(char const *name, HyPerCol *hc);

   virtual ~MomentumUpdater() {}

   char const *getMomentumMethod() { return mMomentumMethod; }

  protected:
   MomentumUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual int updateWeights(int arborId) override;

   void applyMomentum(int arborId);

   void applyMomentum(int arborId, float dwFactor, float wFactor);

  protected:
   enum Method { UNDEFINED_METHOD, SIMPLE, VISCOSITY, ALEX };

   char *mMomentumMethod = nullptr;
   Method mMethod        = UNDEFINED_METHOD;
   float mMomentumTau    = 0.25f;
   float mMomentumDecay  = 0.0f;

   Weights *mPrevDeltaWeights = nullptr;
};

} // namespace PV

#endif // MOMENTUMUPDATER_HPP_
