/*
 * NormalizeBase.hpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEBASE_HPP_
#define NORMALIZEBASE_HPP_

#include "../connections/HyPerConn.hpp"
#include <assert.h>

namespace PV {

class NormalizeBase {
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

   const char * getName() {return name;}
   const float getStrength() {return strength;}
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
   const bool  getSymmetrizeWeightsFlag() {return symmetrizeWeightsFlag;}
#endif // OBSOLETE
   const bool  getNormalizeArborsIndividuallyFlag() {return normalizeArborsIndividually;}

protected:
   NormalizeBase();
   int initialize(const char * name, HyPerCol * hc);

   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
   virtual void ioParam_symmetrizeWeights(enum ParamsIOFlag ioFlag);
#endif // OBSOLETE
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
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
   int symmetrizeWeights(HyPerConn * conn); // may be used by several subclasses
#endif // OBSOLETE
   static void normalizePatch(pvwdata_t * dataStart, int weights_per_patch, float multiplier);
   HyPerCol * parent() { return parentHyPerCol; }

private:
   int initialize_base();

// Member variables
protected:
   char * name;
   HyPerCol * parentHyPerCol;
   HyPerConn ** connectionList;
   int numConnections;
   float strength;                    // Value to normalize to; precise interpretation depends on normalization method
   // normalizeFromPostPerspective,rMinX,rMinY,normalize_cutoff moved to NormalizeMultiply
#ifdef OBSOLETE // Marked obsolete Oct 24, 2014.  symmetrizeWeights is too specialized for NormalizeBase.  Create a new subclass to restore this functionality
   bool symmetrizeWeightsFlag;        // Whether to call symmetrizeWeights.  Only meaningful if pre->nf==post->nf and connection is one-to-one
#endif // OBSOLETE

   bool normalizeArborsIndividually;  // If true, each arbor is treated as its own connection.  If false, each patch groups all arbors together and normalizes them in common.

   bool normalizeOnInitialize;        // Whether to normalize weights when setting the weights' initial values
   bool normalizeOnWeightUpdate;      // Whether to normalize weights when the weights have been updated
}; // end of class NormalizeBase

} // end namespace PV

#endif /* NORMALIZEBASE_HPP_ */
