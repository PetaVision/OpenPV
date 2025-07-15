/*
 * SharedWeights.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTS_HPP_
#define SHAREDWEIGHTS_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the sharedWeights flag from parameters.
 * patch size. The dimensions are read from the sharedWeights parameter, and
 * retrieved using the getSharedWeights() method.
 */
class SharedWeights : public BaseObject {
  protected:
   /**
    * List of parameters needed from the SharedWeights class
    * @name SharedWeights Parameters
    * @{
    */

   /**
    * @brief sharedWeights: Boolean, defines if the weights use shared weights or not.
    * Defaults to true (shared weights).
    */
   virtual void ioParam_sharedWeights(ParamsIOSwitch ioSwitch);

   /** @} */ // end of SharedWeights parameters

  public:
   SharedWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~SharedWeights();

   bool getSharedWeightsFlag() const { return mSharedWeightsFlag; }

  protected:
   SharedWeights() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   bool mSharedWeightsFlag = true;
};

} // namespace PV

#endif // SHAREDWEIGHTS_HPP_
