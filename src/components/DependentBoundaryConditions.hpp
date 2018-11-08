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
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief valueBC: Not used by DependentBoundaryConditions; instead the valueBC
    * parameter is copied from the layer named by the OriginalLayerNameParam parameter.
    */
   virtual void ioParam_valueBC(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief
    */

   /** @} */ // end of DependentBoundaryConditions parameters

  public:
   DependentBoundaryConditions(char const *name, PVParams *params, Communicator *comm);

   virtual ~DependentBoundaryConditions();

  protected:
   DependentBoundaryConditions() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // DEPENDENTBOUNDARYCONDITIONS_HPP_
