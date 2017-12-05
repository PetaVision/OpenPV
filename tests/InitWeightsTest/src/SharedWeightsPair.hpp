/*
 * SharedWeightsPair.hpp
 *
 *  Created on: Dec 4, 2017
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTSPAIR_HPP_
#define SHAREDWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

class SharedWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the SharedWeightsPair class
    * @name SharedWeightsPair Parameters
    * @{
    */

   /**
    * @brief sharedWeights: SharedWeightsPair always sets sharedWeights to true
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   /** @} */ // end of SharedWeightsPair parameters

  public:
   SharedWeightsPair(char const *name, HyPerCol *hc);

   virtual ~SharedWeightsPair() {}

  protected:
   SharedWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;
};

} // namespace PV

#endif // SHAREDWEIGHTSPAIR_HPP_
