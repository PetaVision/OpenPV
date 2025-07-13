/*
 * DependentBoundaryConditions.hpp
 *
 *  Created on: Jul 30, 2018
 *      Author: Pete Schultz
 */

#ifndef DEPENDENTBOUNDARYCONDITIONS_HPP_
#define DEPENDENTBOUNDARYCONDITIONS_HPP_

#include "components/BoundaryConditions.hpp"

namespace PV {

/**
 * A component to use the same phase as another BoundaryConditions object,
 * named in the originalLayerName parameter.
 */
class DependentBoundaryConditions : public BoundaryConditions {
  protected:
   /**
    * List of parameters needed from the DependentBoundaryConditions class
    * @name DependentBoundaryConditions Parameters
    * @{
    */

   /**
    * @brief mirrorBCflag: Not used by DependentBoundaryConditions; instead the flag
    * is copied from the layer named by the OriginalLayerNameParam parameter.
    */
   virtual void ioParam_mirrorBCflag(ParamsIOSwitch ioSwitch) override;

   /**
    * @brief valueBC: Not used by DependentBoundaryConditions; instead the valueBC
    * parameter is copied from the layer named by the OriginalLayerNameParam parameter.
    */
   virtual void ioParam_valueBC(ParamsIOSwitch ioSwitch) override;

   /**
    * @brief
    */

   /** @} */ // end of DependentBoundaryConditions parameters

  public:
   DependentBoundaryConditions(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~DependentBoundaryConditions();

  protected:
   DependentBoundaryConditions() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // DEPENDENTBOUNDARYCONDITIONS_HPP_
