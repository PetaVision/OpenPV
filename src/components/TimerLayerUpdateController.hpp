/*
 * TimerLayerUpdateController.hpp
 */

#ifndef TIMERLAYERUPDATECONTROLLER_HPP_
#define TIMERLAYERUPDATECONTROLLER_HPP_

#include "columns/Communicator.hpp"
#include "components/LayerUpdateController.hpp"
#include "io/PVParams.hpp"

namespace PV {

/**
 * A LayerUpdateController component to cause needUpdate to return true periodically, with
 * period given by the timerPeriod parameter. This allows for periodic triggering.
 */
class TimerLayerUpdateController : public LayerUpdateController {
  protected:
   /**
    * List of parameters needed from the TimerLayerUpdateController class
    * @name TimerLayer Parameters
    * @{
    */

   /**
    * @brief triggerLayerName: TimerLayerUpdateController does not use the triggerLayerName param
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief timerPeriod: The time interval between timesteps where needUpdate returns true
    */
   virtual void ioParam_timerPeriod(enum ParamsIOFlag ioFlag);

   /** @} */ // end of TimerLayerUpdateController parameters

  public:

   TimerLayerUpdateController(char const *name, PVParams *params, Communicator const *comm);
   virtual ~TimerLayerUpdateController();

   /**
     * A virtual function to determine if the layer will update on the given timestep.
     * Default behavior is dependent on the triggering method.
     * If there is triggering with trigger behavior updateOnlyOnTrigger, returns
     * the trigger layer's needUpdate for the time simTime + triggerOffset.
     * Otherwise, returns true if simTime is LastUpdateTime, LastUpdateTime + getDeltaUpdateTime(),
     * LastUpdateTime + 2*getDeltaUpdateTime(), LastUpdateTime + 3*getDeltaUpdateTime(), etc.
     * @return Returns true if an update is needed on that timestep, false otherwise.
     */
   virtual bool needUpdate(double simTime, double deltaTime) const override;

   double getTimerPeriod() const { return mDeltaUpdateTime; }

  protected:
   TimerLayerUpdateController();

   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * This routine initializes the InternalStateBuffer and ActivityBuffer components. It also sets
    * the LastUpdateTime data member to the DeltaTime argument of the message.
    * (The reason for doing so is that if the layer updates every 10th timestep, it generally
    * should update on timesteps 1, 11, 21, etc.; not timesteps 0, 10, 20, etc.
    * InitializeState is the earliest message that passes the HyPerCol's DeltaTime argument.)
    */
   Response::Status initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * A virtual method, called by initializeState() to set the interval between times when
    * updateActivity is needed, if the layer does not have a trigger layer. If the layer does have
    * a trigger layer, this method will not be called and the period is set (during InitializeState)
    * to the that layer's DeltaUpdateTime.
    */
   virtual void setNontriggerDeltaUpdateTime(double deltaTime) override {}

   virtual void applyTrigger(double simTime, double deltaTime) override {}

  protected:
   // No data members specific to this class. the timerPeriod parameter is stored in
   // mDeltaUpdateTime, which is a LayerUpdateController object.

}; // class TimerLayerUpdateController

} // namespace PV

#endif // TIMERLAYERUPDATECONTROLLER_HPP_
