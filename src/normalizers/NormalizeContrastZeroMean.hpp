/*
 * NormalizeContrastZeroMean.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZECONTRASTZEROMEAN_HPP_
#define NORMALIZECONTRASTZEROMEAN_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeContrastZeroMean: public PV::NormalizeBase {
   // Member functions
public:
   NormalizeContrastZeroMean(const char * name, PVParams * params);
   virtual ~NormalizeContrastZeroMean();

   virtual int normalizeWeights(HyPerConn * conn);

protected:
   NormalizeContrastZeroMean();
   int initialize(const char * name, PVParams * params);
   virtual int setParams();

   virtual void readMinSumTolerated() {minSumTolerated = params->value(name, "minSumTolerated", 0.0f, true/*warnIfAbsent*/);}
   virtual void readNormalizeFromPostPerspective() {return;}

   static void subtractOffsetAndNormalize(pvdata_t * dataStartPatch, int weights_per_patch, float offset, float normalizer);
   int accumulateSumAndSumSquared(pvdata_t * dataPatchStart, int weights_in_patch, double * sum, double * sumsq);

private:
   int initialize_base();

   // Member variables
protected:
   float minSumTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

} /* namespace PV */
#endif /* NORMALIZECONTRASTZEROMEAN_HPP_ */
