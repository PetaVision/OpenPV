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
   virtual void ioParam_sharedWeights(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of SharedWeightsTrue parameters

  public:
   SharedWeightsTrue(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~SharedWeightsTrue();

  protected:
   SharedWeightsTrue() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
};

} // namespace PV

#endif // SHAREDWEIGHTSTRUE_HPP_
