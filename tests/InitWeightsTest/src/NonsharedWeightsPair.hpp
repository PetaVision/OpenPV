/*
 * NonsharedWeightsPair.hpp
 *
 *  Created on: Dec 4, 2017
 *      Author: Pete Schultz
 */

#ifndef NONSHAREDWEIGHTSPAIR_HPP_
#define NONSHAREDWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

class NonsharedWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the NonsharedWeightsPair class
    * @name NonsharedWeightsPair Parameters
    * @{
    */

   /**
    * @brief sharedWeights: NonsharedWeightsPair always sets sharedWeights to true
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   /** @} */ // end of NonsharedWeightsPair parameters

  public:
   NonsharedWeightsPair(char const *name, HyPerCol *hc);

   virtual ~NonsharedWeightsPair() {}

  protected:
   NonsharedWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;
};

} // namespace PV

#endif // NONSHAREDWEIGHTSPAIR_HPP_
