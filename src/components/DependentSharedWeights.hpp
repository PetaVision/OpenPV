/*
 * DependentSharedWeights.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef DEPENDENTSHAREDWEIGHTS_HPP_
#define DEPENDENTSHAREDWEIGHTS_HPP_

#include "components/SharedWeights.hpp"

namespace PV {

/**
 * A subclass of SharedWeights, which retrieves the sharedWeights flag from the connection
 * named in an OriginalConnNameParam component, instead of reading it from params.
 */
class DependentSharedWeights : public SharedWeights {
  protected:
   /**
    * List of parameters needed from the DependentSharedWeights class
    * @name DependentSharedWeights Parameters
    * @{
    */

   /**
    * @brief shareeWeihgts: DependentSharedWeightss does not use the sharedWeights parameter,
    * but uses the same setting as the original connection.
    */
   virtual void ioParam_sharedWeights(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of DependentSharedWeights parameters

  public:
   DependentSharedWeights(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~DependentSharedWeights();

  protected:
   DependentSharedWeights();

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

}; // class DependentSharedWeights

} // namespace PV

#endif // DEPENDENTSHAREDWEIGHTS_HPP_
