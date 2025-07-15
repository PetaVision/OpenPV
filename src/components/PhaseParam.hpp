/*
 * PhaseParam.hpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#ifndef PHASEPARAM_HPP_
#define PHASEPARAM_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the phase parameter from the params file.
 */
class PhaseParam : public BaseObject {
  protected:
   /**
    * List of parameters needed from the PhaseParam class
    * @name PhaseParam Parameters
    * @{
    */

   /**
    * @brief phase: specifies the value of the phase parameter. Layers use the
    * phase parameter to control which layers must get updated previous to which
    * other layers.
    */
   virtual void ioParam_phase(ParamsIOSwitch ioSwitch);

   /** @} */ // end of PhaseParam parameters

  public:
   PhaseParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~PhaseParam();

   Response::Status respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message);

   int getPhase() const { return mPhase; }

  protected:
   PhaseParam() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status setMaxPhase(int *maxPhase);

  protected:
   int mPhase = 0;
};

} // namespace PV

#endif // PHASEPARAM_HPP_
