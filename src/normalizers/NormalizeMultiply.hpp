/*
 * NormalizeMultiply.hpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#ifndef NORMALIZEMULTIPLY_HPP_
#define NORMALIZEMULTIPLY_HPP_

#include "NormalizeBase.hpp"
#include "components/Weights.hpp"

namespace PV {

class NormalizeMultiply : public NormalizeBase {
   // Member functions
  protected:
   /**
    * List of parameters needed from the NormalizeMultiply class
    * @name NormalizeMultiply Parameters
    * @{
    */

   /**
    * Sets the size in the x-direction of the rectangle zeroed out by applyRMin
    */
   virtual void ioParam_rMinX(enum ParamsIOFlag ioFlag);

   /**
    * Sets the size in the y-direction of the rectangle zeroed out by applyRMin
    */
   virtual void ioParam_rMinY(enum ParamsIOFlag ioFlag);

   /**
    * If set to true, negative weights are replaced by zero
    */
   virtual void ioParam_nonnegativeConstraintFlag(enum ParamsIOFlag ioFlag);

   /**
    * If positive, all weights whose absolute value is less than
    * (normalize_cutoff * (max weight over all patches and all arbors))
    * are set to zero.  The maximum weight is calculated after applying
    * the behavior defined by rMinX, rMinY and nonnegativeConstraintFlag.
    */
   virtual void ioParam_normalize_cutoff(enum ParamsIOFlag ioFlag);

   /**
    * If set to true, the weights are group based on the index of the post-synaptic neuron.
    * If false, the index of the pre-synaptic neuron is used.
    *
    * Currently only meaningfull if sharedWeights is true and normalizeMethod is
    * normalizeSum or normalizeL2.
    */
   virtual void ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag);
   // If false, group all weights with a common presynaptic
   // neuron for normalizing.  If true, group all weights with a
   // common postsynaptic neuron
   // Only meaningful (at least for now) for KernelConns using sum of weights or sum of squares
   // normalization methods.

   /** @} */ // end of NormalizeMultiply parameters

  public:
   NormalizeMultiply(const char *name, HyPerCol *hc);
   virtual ~NormalizeMultiply();

   float getRMinX() { return mRMinX; }
   float getRMinY() { return mRMinY; }
   float getNormalizeCutoff() { return mNormalizeCutoff; }
   bool getNormalizeFromPostPerspectiveFlag() { return mNormalizeFromPostPerspective; }

   virtual int normalizeWeights() override;

  protected:
   NormalizeMultiply();
   int initialize(const char *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   int applyThreshold(
         float *dataPatchStart,
         int weights_in_patch,
         float wMax); // weights less than normalize_cutoff*max(weights) are zeroed out

   /**
    * Zeroes out all weights in a rectangle defined by the member variables rMinX, rMinY.
    * dataPatchStart is a pointer to a patch.
    * nxp and nyp are the dimensions of the patch
    * xPatchStride, yPatchStride are the strides in the x- and y-directions.  nfp is implicitly the
    * same as xPatchStride
    * rMinX and rMinY are the half-width and half-height of the rectangle to be zeroed out.
    * The rectangle to be zeroed out is centered at the center of the patch.
    * Pixels in the interior of the rectangle are set to zero, but pixels exactly on the edge of the
    * rectangle are not changed.
    */
   int applyRMin(
         float *dataPatchStart,
         float rMinX,
         float rMinY,
         int nxp,
         int nyp,
         int xPatchStride,
         int yPatchStride);

   static void normalizePatch(float *patchData, int weightsPerPatch, float multiplier);

   // Member variables
  protected:
   float mRMinX                       = 0.0f;
   float mRMinY                       = 0.0f;
   bool mNonnegativeConstraintFlag    = false;
   float mNormalizeCutoff             = 0.0f;
   bool mNormalizeFromPostPerspective = false;
}; // class NormalizeMultiply

} /* namespace PV */

#endif /* NORMALIZEMULTIPLY_HPP_ */
