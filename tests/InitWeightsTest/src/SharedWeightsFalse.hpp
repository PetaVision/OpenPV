/*
 * SharedWeightsFalse.hpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTSFALSE_HPP_
#define SHAREDWEIGHTSFALSE_HPP_

#include "components/SharedWeights.hpp"

namespace PV {

/**
 * A derived class of SharedWeights that always sets the flag to false.
 */
class SharedWeightsFalse : public SharedWeights {
  protected:
   /**
    * List of parameters needed from the SharedWeightsFalse class
    * @name SharedWeightsFalse Parameters
    * @{
    */

   /**
    * @brief sharedWeights: SharedWeightsFalse always sets the sharedWeights flag to false.
    * Defaults to false (non-shared weights).
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of SharedWeightsFalse parameters

  public:
   SharedWeightsFalse(char const *name, HyPerCol *hc);

   virtual ~SharedWeightsFalse();

  protected:
   SharedWeightsFalse() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
};

} // namespace PV

#endif // SHAREDWEIGHTSFALSE_HPP_
