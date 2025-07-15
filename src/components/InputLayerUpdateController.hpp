/*
 * InputLayerUpdateController.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: pschultz
 */

#ifndef INPUTLAYERUPDATECONTROLLER_HPP_
#define INPUTLAYERUPDATECONTROLLER_HPP_

#include "components/LayerUpdateController.hpp"

namespace PV {

/**
 * A component to determine if a layer should update on the current timestep, and to handle
 * triggering behavior.
 */
class InputLayerUpdateController : public LayerUpdateController {
  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name InputLayer Parameters
    * @{
    */

   /**
    * triggerLayerName: InputLayer and derived classes do not use triggering, and always set
    * triggerLayerName to NULL.
    */
   virtual void ioParam_triggerLayerName(ParamsIOSwitch ioSwitch) override;
   /** @} */ // end of InputLayerUpdateController parameters

  public:
   InputLayerUpdateController(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~InputLayerUpdateController();

  protected:
   InputLayerUpdateController();

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   /**
    * DeltaUpdateTime for InputLayer classes is the displayPeriod param, if nonzero,
    * and MAX_DBL if displayPeriod == 0.
    */
   virtual void setNontriggerDeltaUpdateTime(double deltaTime) override;

  protected:
}; // class InputLayerUpdateController

} // namespace PV

#endif // INPUTLAYERUPDATECONTROLLER_HPP_
