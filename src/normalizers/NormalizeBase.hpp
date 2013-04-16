/*
 * NormalizeBase.hpp
 *
 *  Created on: Apr 5, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEBASE_HPP_
#define NORMALIZEBASE_HPP_

#include "../connections/HyPerConn.hpp"
#include "../connections/KernelConn.hpp"
#include <assert.h>

namespace PV {

class NormalizeBase {
// Member functions
public:
   // no public method; only subclasses can be constructed directly
   virtual ~NormalizeBase() = 0;

   virtual int normalizeWeights(HyPerConn * conn);

protected:
   NormalizeBase();
   int initialize(const char * name, PVParams * params);
   virtual int setParams();

   virtual void readStrength() {strength = params->value(name, "strength", 1.0f, true/*warnIfAbsent*/);}
   virtual void readNormalizeCutoff() {normalize_cutoff = params->value(name, "normalize_cutoff", 0.0f, true/*warnIfAbsent*/);}
   virtual void readSymmetrizeWeights() {symmetrizeWeightsFlag = params->value(name, "symmetrizeWeights", false/*default value*/, true/*warnIfAbsent*/);}
   virtual void readNormalizeFromPostPerspective() {normalizeFromPostPerspective = params->value(name, "normalizeFromPostPerspective", false/*default value*/, true/*warnIfAbsent*/);}
   virtual void readNormalizeArborsIndividually() {normalizeArborsIndividually = params->value(name, "normalizeArborsIndividually", false/*default value*/, true/*warnIfAbsent*/);}

   int accumulateSum(pvdata_t * dataPatchStart, int weights_in_patch, double * sum);
   int accumulateSumSquared(pvdata_t * dataPatchStart, int weights_in_patch, double * sumsq);
   int accumulateMax(pvdata_t * dataPatchStart, int weights_in_patch, float * max);
   int applyThreshold(pvdata_t * dataPatchStart, int weights_in_patch, float wMax); // weights less than normalize_cutoff*max(weights) are zeroed out
   int symmetrizeWeights(HyPerConn * conn); // may be used by several subclasses
   static void normalizePatch(pvdata_t * dataStart, int weights_per_patch, float multiplier);

private:
   int initialize_base();

// Member variables
protected:
   char * name;
   PVParams * params;
   float strength;                    // Value to normalize to; precise interpretation depends on normalization method
   float normalize_cutoff;            // If true, weights with abs(w)<max(abs(w))*normalize_cutoff are truncated to zero.
   bool symmetrizeWeightsFlag;        // Whether to call symmetrizeWeights.  Only meaningful if pre->nf==post->nf and connection is one-to-one
   bool normalizeFromPostPerspective; // If false, group all weights with a common presynaptic neuron for normalizing.  If true, group all weights with a common postsynaptic neuron
                                      // Only meaningful (at least for now) for KernelConns using sum of weights or sum of squares normalization methods.

   bool normalizeArborsIndividually;  // If true, each arbor is treated as its own connection.  If false, each patch groups all arbors together and normalizes them in common.
}; // end of class NormalizeBase

} // end namespace PV

#endif /* NORMALIZEBASE_HPP_ */
