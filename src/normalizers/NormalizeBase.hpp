/*
 * NormalizeBase.hpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEBASE_HPP_
#define NORMALIZEBASE_HPP_

#include <assert.h>
#include <columns/BaseObject.hpp>
#include <connections/HyPerConn.hpp>

namespace PV {

class NormalizeBase : public BaseObject {
   // Member functions
  public:
   // no public constructor; only subclasses can be constructed directly
   virtual ~NormalizeBase() = 0;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * Appends the indicated connection to the list of connections for this normalizer
    */
   int addConnToList(HyPerConn *newConn);

   /**
    * Called by HyPerConn::communicateInitInfo this virtual method is where
    * settings that depend on other objects take place.
    * In particular, NormalizeGroup calls its group head's addConnToList
    * method from NormalizeGroup::communicateInitInfo method.
    */
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * The public interface for normalizing weights.
    * If normalizeOnInitialize is true and the simulation time is startTime(),
    * or if normalizeOnWeightUpdate is true and the simulation time is the conn's lastUpdateTime,
    * it calls the (virtual protected) method normalizeWeights
    */
   int normalizeWeightsWrapper();

   float getStrength() const { return strength; }
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
   bool getNormalizeArborsIndividuallyFlag() const { return normalizeArborsIndividually; }

  protected:
   NormalizeBase();
   int initialize(const char *name, HyPerCol *hc);
   virtual int setDescription() override;

   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag);

   virtual int normalizeWeights();
   int accumulateSum(float *dataPatchStart, int weights_in_patch, float *sum);
   int accumulateSumShrunken(
         float *dataPatchStart,
         float *sum,
         int nxpShrunken,
         int nypShrunken,
         int offsetShrunken,
         int xPatchStride,
         int yPatchStride);
   int accumulateSumSquared(float *dataPatchStart, int weights_in_patch, float *sumsq);
   int accumulateSumSquaredShrunken(
         float *dataPatchStart,
         float *sumsq,
         int nxpShrunken,
         int nypShrunken,
         int offsetShrunken,
         int xPatchStride,
         int yPatchStride);
   int accumulateMaxAbs(float *dataPatchStart, int weights_in_patch, float *max);
   int accumulateMax(float *dataPatchStart, int weights_in_patch, float *max);
   int accumulateMin(float *dataPatchStart, int weights_in_patch, float *max);
   static void normalizePatch(float *dataStart, int weights_per_patch, float multiplier);

  private:
   int initialize_base();

   // Member variables
  protected:
   std::vector<HyPerConn *> connectionList;
   float strength; // Value to normalize to; precise interpretation depends on normalization method

   bool normalizeArborsIndividually; // If true, each arbor is treated as its own connection.  If
   // false, each patch groups all arbors together and normalizes
   // them in common.

   bool normalizeOnInitialize; // Whether to normalize weights when setting the weights' initial
   // values
   bool normalizeOnWeightUpdate; // Whether to normalize weights when the weights have been updated
}; // end of class NormalizeBase

} // end namespace PV

#endif /* NORMALIZEBASE_HPP_ */
