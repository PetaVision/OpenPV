/*
 * SharedWeightsTrue.hpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTSTRUE_HPP_
#define SHAREDWEIGHTSTRUE_HPP_

#include "components/SharedWeights.hpp"

namespace PV {

/**
 * A derived class of SharedWeights that always sets the flag to false.
 */
class SharedWeightsTrue : public SharedWeights {
  protected:
   /**
    * List of parameters needed from the SharedWeightsTrue class
    * @name SharedWeightsTrue Parameters
    * @{
    */

   /**
    * @brief sharedWeights: SharedWeightsTrue always sets the sharedWeights flag to false.
    * Defaults to false (non-shared weights).
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of SharedWeightsTrue parameters

  public:
   SharedWeightsTrue(char const *name, HyPerCol *hc);

   virtual ~SharedWeightsTrue();

  protected:
   SharedWeightsTrue() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
};

} // namespace PV

#endif // SHAREDWEIGHTSTRUE_HPP_
