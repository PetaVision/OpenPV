/*
 * DependentPhaseParam.hpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#ifndef DEPENDENTPHASEPARAM_HPP_
#define DEPENDENTPHASEPARAM_HPP_

#include "components/PhaseParam.hpp"

namespace PV {

/**
 * A component to use the same phase as another PhaseParam object,
 * named in the originalLayerName parameter.
 */
class DependentPhaseParam : public PhaseParam {
  protected:
   /**
    * List of parameters needed from the DependentPhaseParam class
    * @name DependentPhaseParam Parameters
    * @{
    */

   /**
    * @brief phase: Not used by DependentPhaseParam; instead the phase
    * is copied from the layer named by the OriginalLayerNameParam parameter.
    */
   virtual void ioParam_phase(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief
    */

   /** @} */ // end of DependentPhaseParam parameters

  public:
   DependentPhaseParam(char const *name, HyPerCol *hc);

   virtual ~DependentPhaseParam();

  protected:
   DependentPhaseParam() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // DEPENDENTPHASEPARAM_HPP_
