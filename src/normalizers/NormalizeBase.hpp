/*
 * NormalizeBase.hpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEBASE_HPP_
#define NORMALIZEBASE_HPP_

#include <columns/BaseObject.hpp>
#include <connections/HyPerConn.hpp>
#include <assert.h>

namespace PV {

class NormalizeBase : public BaseObject {
// Member functions
public:
   // no public constructor; only subclasses can be constructed directly
   virtual ~NormalizeBase() = 0;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * Appends the indicated connection to the list of connections for this normalizer
    */
   int addConnToList(HyPerConn * newConn);

   /**
    * The public interface for normalizing weights.
    * If normalizeOnInitialize is true and the simulation time is startTime(),
    * or if normalizeOnWeightUpdate is true and the simulation time is the conn's lastUpdateTime,
    * it calls the (virtual protected) method normalizeWeights
    */
   int normalizeWeightsWrapper();

   const float getStrength() {return strength;}
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
   const bool  getNormalizeArborsIndividuallyFlag() {return normalizeArborsIndividually;}

protected:
   NormalizeBase();
   int initialize(const char * name, HyPerCol * hc);

   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag);

   virtual int normalizeWeights();
   int accumulateSum(pvwdata_t * dataPatchStart, int weights_in_patch, double * sum);
   int accumulateSumShrunken(pvwdata_t * dataPatchStart, double * sum,
   		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride);
   int accumulateSumSquared(pvwdata_t * dataPatchStart, int weights_in_patch, double * sumsq);
   int accumulateSumSquaredShrunken(pvwdata_t * dataPatchStart, double * sumsq,
   		int nxpShrunken, int nypShrunken, int offsetShrunken, int xPatchStride, int yPatchStride);
   int accumulateMaxAbs(pvwdata_t * dataPatchStart, int weights_in_patch, float * max);
   int accumulateMax(pvwdata_t * dataPatchStart, int weights_in_patch, float * max);
   int accumulateMin(pvwdata_t * dataPatchStart, int weights_in_patch, float * max);
   static void normalizePatch(pvwdata_t * dataStart, int weights_per_patch, float multiplier);

private:
   int initialize_base();

// Member variables
protected:
   HyPerConn ** connectionList;
   int numConnections;
   float strength;                    // Value to normalize to; precise interpretation depends on normalization method

   bool normalizeArborsIndividually;  // If true, each arbor is treated as its own connection.  If false, each patch groups all arbors together and normalizes them in common.

   bool normalizeOnInitialize;        // Whether to normalize weights when setting the weights' initial values
   bool normalizeOnWeightUpdate;      // Whether to normalize weights when the weights have been updated
}; // end of class NormalizeBase

BaseObject * createNormalizeBase(char const * name, HyPerCol * hc);

} // end namespace PV

#endif /* NORMALIZEBASE_HPP_ */
