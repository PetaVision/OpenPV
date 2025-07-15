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

class NormalizeContrastZeroMean : public NormalizeBase {
   // Member functions
  public:
   NormalizeContrastZeroMean(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~NormalizeContrastZeroMean();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeContrastZeroMean();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void ioParam_minSumTolerated(ParamsIOSwitch ioSwitch);

   static void subtractOffsetAndNormalize(
         float *dataStartPatch,
         int weightsPerPatch,
         float offset,
         float normalizer);
   int accumulateSumAndSumSquared(
         float *dataPatchStart,
         int weights_in_patch,
         float *sum,
         float *sumsq);

  private:
   int initialize_base();

   // Member variables
  protected:
   float minSumTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
}; // class NormalizeContrastZeroMean

} /* namespace PV */
#endif /* NORMALIZECONTRASTZEROMEAN_HPP_ */
