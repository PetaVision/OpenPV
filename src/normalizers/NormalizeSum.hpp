/*
 * NormalizeSum.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZESUM_HPP_
#define NORMALIZESUM_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeSum : public NormalizeMultiply {
   // Member functions
  public:
   NormalizeSum(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~NormalizeSum();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeSum();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void ioParam_minSumTolerated(ParamsIOSwitch ioSwitch);

  private:
   int initialize_base();

   // Member variables
  protected:
   float mMinSumTolerated = 0.0f; // Error if any patch has abs(sum(weights)) less than this amount.

}; // class NormalizeSum

} /* namespace PV */
#endif /* NORMALIZESUM_HPP_ */
