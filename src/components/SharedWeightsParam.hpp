/*
 * SharedWeightsParam.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTSPARAM_HPP_
#define SHAREDWEIGHTSPARAM_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the sharedWeights flag from parameters.
 * patch size. The dimensions are read from the sharedWeights parameter, and
 * retrieved using the getSharedWeights() method.
 */
class SharedWeightsParam : public BaseObject {
  protected:
   /**
    * List of parameters needed from the SharedWeightsParam class
    * @name SharedWeightsParam Parameters
    * @{
    */

   /**
    * @brief sharedWeights: Boolean, defines if the weights use shared weights or not.
    * Defaults to false (non-shared weights).
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);

   /** @} */ // end of SharedWeightsParam parameters

  public:
   SharedWeightsParam(char const *name, HyPerCol *hc);

   virtual ~SharedWeightsParam();

   bool getSharedWeights() const { return mSharedWeights; }

  protected:
   SharedWeightsParam() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   bool mSharedWeights = false;
};

} // namespace PV

#endif // SHAREDWEIGHTSPARAM_HPP_
