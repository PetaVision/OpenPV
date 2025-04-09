/*
 * InputVolumeLayerUpdateController.hpp
 */

#ifndef INPUTVOLUMELAYERUPDATECONTROLLER_HPP_
#define INPUTVOLUMELAYERUPDATECONTROLLER_HPP_

#include "components/LayerUpdateController.hpp"

namespace PV {

/**
 * A component to determine if a layer should update on the current timestep, and to handle
 * triggering behavior.
 */
class InputVolumeLayerUpdateController : public LayerUpdateController {
  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name InputVolumeLayer Parameters
    * @{
    */

   /**
    * triggerLayerName: InputVolumeLayer and derived classes do not use triggering, and always set
    * triggerLayerName to NULL.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   /** @} */ // end of InputVolumeLayerUpdateController parameters

  public:
   InputVolumeLayerUpdateController(char const *name, PVParams *params, Communicator const *comm);
   virtual ~InputVolumeLayerUpdateController();

  protected:
   InputVolumeLayerUpdateController();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   /**
    * DeltaUpdateTime for InputVolumeLayer classes is the displayPeriod param, if nonzero,
    * and MAX_DBL if displayPeriod == 0.
    */
   virtual void setNontriggerDeltaUpdateTime(double deltaTime) override;

  protected:
}; // class InputVolumeLayerUpdateController

} // namespace PV

#endif // INPUTVOLUMELAYERUPDATECONTROLLER_HPP_
