/*
 * ImpliedWeightsPair.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef IMPLIEDWEIGHTSPAIR_HPP_
#define IMPLIEDWEIGHTSPAIR_HPP_

#include "components/WeightsPairInterface.hpp"

namespace PV {

class ImpliedWeightsPair : public WeightsPairInterface {
  public:
   ImpliedWeightsPair(char const *name, HyPerCol *hc);

   virtual ~ImpliedWeightsPair();

  protected:
   ImpliedWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   virtual void createPreWeights() override;
   virtual void createPostWeights() override;
};

} // namespace PV

#endif // IMPLIEDWEIGHTSPAIR_HPP_
