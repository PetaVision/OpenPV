/*
 * NormalizeBase.hpp
 *
 *  Created on: Apr 5, 2013
 *      Author: Pete Schultz
 */

#ifndef NORMALIZEBASE_HPP_
#define NORMALIZEBASE_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "components/Weights.hpp"

namespace PV {

class NormalizeBase : public BaseObject {
  protected:
   /**
    * List of parameters needed from the NormalizeBase class
    * @name NormalizeBase Parameters
    * @{
    */
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag);
   /** @} */ // end of NormalizeBase parameters

  public:
   NormalizeBase(char const *name, HyPerCol *hc);

   virtual ~NormalizeBase() {}

   /**
    * If the normalizeOnInitialize flag is set and the simulation time is startTime(),
    * or if the normalizeOnWeightUpdate flag is set and the weight updater's LastUpdateTime
    * is greater than the normalizer's LastUpdateTime, this method calls the (virtual protected)
    * method normalizeWeights(). Otherwise, this method does nothing.
    */
   void normalizeWeightsIfNeeded();

   float getStrength() const { return mStrength; }
   bool getNormalizeArborsIndividuallyFlag() const { return mNormalizeArborsIndividually; }
   bool getNormalizeOnInitialize() const { return mNormalizeOnInitialize; }
   bool getNormalizeOnWeightUpdate() const { return mNormalizeOnWeightUpdate; }

  protected:
   NormalizeBase() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual int normalizeWeights() { return PV_SUCCESS; }

   virtual bool weightsHaveUpdated() { return false; }

  protected:
   float mStrength = 1.0f;

   bool mNormalizeArborsIndividually = false;
   bool mNormalizeOnInitialize       = true;
   bool mNormalizeOnWeightUpdate     = true;

   double mLastUpdateTime = 0.0;
};

} // namespace PV

#endif // NORMALIZEBASE_HPP_
