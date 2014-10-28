/*
 * NormalizeMultiply.hpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#ifndef NORMALIZEMULTIPLY_HPP_
#define NORMALIZEMULTIPLY_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeMultiply: public NormalizeBase {
// Member functions
public:
   NormalizeMultiply(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections);
   virtual ~NormalizeMultiply();

   const char * getName() { return name; }
   const float getRMinX() { return rMinX; }
   const float getRMinY() { return rMinY; }
   const float getNormalizeCutoff() { return normalize_cutoff; }
   const bool  getNormalizeFromPostPerspectiveFlag() {return normalizeFromPostPerspective;}

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_rMinX(enum ParamsIOFlag ioFlag);
   virtual void ioParam_rMinY(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalize_cutoff(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag);

   virtual int normalizeWeights();

protected:
   NormalizeMultiply();
   int initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections);

   int applyThreshold(pvwdata_t * dataPatchStart, int weights_in_patch, float wMax); // weights less than normalize_cutoff*max(weights) are zeroed out

   /**
    * Zeroes out all weights in a rectangle defined by the member variables rMinX, rMinY.
    * dataPatchStart is a pointer to a patch.
    * nxp and nyp are the dimensions of the patch
    * xPatchStride, yPatchStride are the strides in the x- and y-directions.  nfp is implicitly the same as xPatchStride
    * rMinX and rMinY are the half-width and half-height of the rectangle to be zeroed out.
    * The rectangle to be zeroed out is centered at the center of the patch.
    * Pixels in the interior of the rectangle are set to zero, but pixels exactly on the edge of the rectangle are not changed.
    */
   int applyRMin(pvwdata_t * dataPatchStart, float rMinX, float rMinY,
            int nxp, int nyp, int xPatchStride, int yPatchStride);


private:
   int initialize_base();

// Member variables
protected:
   float rMinX, rMinY;                // zero all weights within rectangle rMinxY, rMInY aligned with center of patch
   float normalize_cutoff;            // If true, weights with abs(w)<max(abs(w))*normalize_cutoff are truncated to zero.
   bool normalizeFromPostPerspective; // If false, group all weights with a common presynaptic neuron for normalizing.  If true, group all weights with a common postsynaptic neuron
                                      // Only meaningful (at least for now) for KernelConns using sum of weights or sum of squares normalization methods.
};

} /* namespace PV */

#endif /* NORMALIZEMULTIPLY_HPP_ */
