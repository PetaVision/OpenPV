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

   /**
    * @brief normalizeMethod: Specifies the type of weight normalization.
    * @details This parameter is not used directly inside the NormalizeBase class,
    * except to include the parameter in generated params files.
    * Generally, instantiation should proceed by separately reading the
    * NormalizeMethod and using the Factory::createByKeyword template to
    * instantiate the function.
    */
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag);
   /** @} */ // end of NormalizeBase parameters

  public:
   NormalizeBase(char const *name, HyPerCol *hc);

   virtual ~NormalizeBase() {}

   void addWeightsToList(Weights *weights);
   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   float getStrength() const { return mStrength; }
   bool getNormalizeArborsIndividuallyFlag() const { return mNormalizeArborsIndividually; }
   bool getNormalizeOnInitialize() const { return mNormalizeOnInitialize; }
   bool getNormalizeOnWeightUpdate() const { return mNormalizeOnWeightUpdate; }

  protected:
   NormalizeBase() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * If the normalizeOnInitialize flag is set and the simulation time is startTime(),
    * or if the normalizeOnWeightUpdate flag is set and the weight updater's LastUpdateTime
    * is greater than the normalizer's LastUpdateTime, this method calls the (virtual protected)
    * method normalizeWeights(). Otherwise, this method does nothing.
    */
   Response::Status
   respondConnectionNormalize(std::shared_ptr<ConnectionNormalizeMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   bool weightsHaveUpdated() const;

   virtual int normalizeWeights() { return PV_SUCCESS; }

   static int accumulateSum(float *dataPatchStart, int weights_in_patch, float *sum);
   static int accumulateSumShrunken(
         float *dataPatchStart,
         float *sum,
         int nxpShrunken,
         int nypShrunken,
         int offsetShrunken,
         int xPatchStride,
         int yPatchStride);
   static int accumulateSumSquared(float *dataPatchStart, int weights_in_patch, float *sumsq);
   static int accumulateSumSquaredShrunken(
         float *dataPatchStart,
         float *sumsq,
         int nxpShrunken,
         int nypShrunken,
         int offsetShrunken,
         int xPatchStride,
         int yPatchStride);
   static int accumulateMaxAbs(float *dataPatchStart, int weights_in_patch, float *max);
   static int accumulateMax(float *dataPatchStart, int weights_in_patch, float *max);
   static int accumulateMin(float *dataPatchStart, int weights_in_patch, float *max);

  protected:
   char *mNormalizeMethod            = nullptr;
   float mStrength                   = 1.0f;
   bool mNormalizeArborsIndividually = false;
   bool mNormalizeOnInitialize       = true;
   bool mNormalizeOnWeightUpdate     = true;

   std::vector<Weights *> mWeightsList;
   double mLastTimeNormalized = 0.0;
};

} // namespace PV

#endif // NORMALIZEBASE_HPP_
