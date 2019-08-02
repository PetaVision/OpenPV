/*
 * LayerUpdateController.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: pschultz
 */

#ifndef LAYERUPDATECONTROLLER_HPP_
#define LAYERUPDATECONTROLLER_HPP_

#include "columns/BaseObject.hpp"

#include "components/ActivityComponent.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/PhaseParam.hpp"

namespace PV {

/**
 * A component to determine if a layer should update on the current timestep, and to handle
 * triggering behavior.
 */
class LayerUpdateController : public BaseObject {
  protected:
   /**
    * List of parameters needed from the LayerUpdateController class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief triggerFlag: triggerFlag is obsolete. If triggerLayerName is NULL or the empty string,
    * triggering is not used. If triggerLayerName is set to a nonempty string, triggering is used.
    */
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: Specifies the name of the layer that this layer triggers off of.
    * If set to NULL or the empty string, the layer does not trigger but updates its state on every
    * timestep.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   // TODO: triggerOffset is measured in units of simulation time, not timesteps.  How does
   // adaptTimeStep affect the triggering time?
   /**
    * @brief triggerOffset: If triggerLayer is set, triggers \<triggerOffset\> timesteps before
    * target trigger
    * @details Defaults to 0
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerBehavior: If triggerLayerName is set, this parameter specifies how the trigger
    * is handled.
    * @details The possible values of triggerBehavior are:
    * - "updateOnlyOnTrigger": updateActivity is called (computing activity buffer from GSyn)
    * only on triggering timesteps.  On other timesteps the layer's state remains unchanged.
    * - "resetStateOnTrigger": On timesteps where the trigger occurs, the membrane potential
    * is copied from the layer specified in triggerResetLayerName and setActivity is called.
    * On nontriggering timesteps, updateActivity is called.
    * For backward compatibility, this parameter defaults to updateOnlyOnTrigger.
    */
   virtual void ioParam_triggerBehavior(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerResetLayerName: If triggerLayerName is set, this parameter specifies the layer
    * to use for updating
    * the state when the trigger happens.  If set to NULL or the empty string, use triggerLayerName.
    */
   virtual void ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of LayerUpdateController parameters

  public:
   enum TriggerBehaviorType { NO_TRIGGER, UPDATEONLYONTRIGGER, RESETSTATEONTRIGGER };

   LayerUpdateController(char const *name, PVParams *params, Communicator const *comm);
   virtual ~LayerUpdateController();

   /**
     * A virtual function to determine if the layer will update on the given timestep.
     * Default behavior is dependent on the triggering method.
     * If there is triggering with trigger behavior updateOnlyOnTrigger, returns
     * the trigger layer's needUpdate for the time simTime + triggerOffset.
     * Otherwise, returns true if simTime is LastUpdateTime, LastUpdateTime + getDeltaUpdateTime(),
     * LastUpdateTime + 2*getDeltaUpdateTime(), LastUpdateTime + 3*getDeltaUpdateTime(), etc.
     * @return Returns true if an update is needed on that timestep, false otherwise.
     */
   virtual bool needUpdate(double simTime, double deltaTime) const;

   double getTriggerOffset() const { return mTriggerOffset; }
   /**
    * A function to return the interval between update times
    */
   double getDeltaUpdateTime() const { return mDeltaUpdateTime; }

   double getLastUpdateTime() { return mLastUpdateTime; }

  protected:
   LayerUpdateController();

   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void setTriggerUpdateController(ObserverTable const *table);
   void setTriggerResetComponent(ObserverTable const *table);

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

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
   virtual void setNontriggerDeltaUpdateTime(double deltaTime);

   Response::Status
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);
   Response::Status
   respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);
   Response::Status respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message);

   virtual void applyTrigger(double simTime, double deltaTime);

  protected:
   bool mTriggerFlag                        = false;
   char *mTriggerLayerName                  = nullptr;
   double mTriggerOffset                    = 0.0;
   char *mTriggerBehavior                   = nullptr;
   TriggerBehaviorType mTriggerBehaviorType = NO_TRIGGER;
   char *mTriggerResetLayerName             = nullptr;

   // Other components within the layer
   PhaseParam *mPhaseParam               = nullptr;
   LayerInputBuffer *mLayerInput         = nullptr;
   ActivityComponent *mActivityComponent = nullptr;

   // Components of other layers used in triggers.
   LayerUpdateController *mTriggerUpdateController = nullptr;
   ActivityComponent *mTriggerResetComponent       = nullptr;

   bool mHasUpdated        = false;
   bool mHasReceived       = false;
   double mDeltaUpdateTime = 1.0;
   double mLastUpdateTime  = 0.0;

}; // class LayerUpdateController

} // namespace PV

#endif // LAYERUPDATECONTROLLER_HPP_
