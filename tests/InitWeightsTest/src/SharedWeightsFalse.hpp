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
   virtual void ioParam_sharedWeights(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of SharedWeightsFalse parameters

  public:
   SharedWeightsFalse(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~SharedWeightsFalse();

  protected:
   SharedWeightsFalse() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
};

} // namespace PV

#endif // SHAREDWEIGHTSFALSE_HPP_
