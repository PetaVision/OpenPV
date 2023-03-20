/*
 * AdaptiveTimeScaleProbe.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESCALEPROBE_HPP_
#define ADAPTIVETIMESCALEPROBE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "components/AdaptiveTimeScaleController.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/AdaptiveTimeScaleProbeOutputter.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/ProbeInterface.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include <memory>

namespace PV {

class AdaptiveTimeScaleProbe : public ProbeInterface {
  protected:
   /**
    * List of parameters needed from the AdaptiveTimeScaleProbe class
    * @name AdaptiveTimeScaleProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the probe that this probe attaches to.
    * The target probe's values are used as the input to compute the adaptive
    * timesteps.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);

   /**
    * @brief baseMax: Specifies the initial maximum timescale allowed.
    * The maximum timescale is allowed to increase at a rate specified
    * by the growthFactor parameter.
    */
   virtual void ioParam_baseMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief baseMin: Specifies the minimum timescale allowed.
    */
   virtual void ioParam_baseMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief tauFactor: If Specifies the coefficient on the effective decay rate
    * used to compute
    * the timescale.
    */
   virtual void ioParam_tauFactor(enum ParamsIOFlag ioFlag);

   /**
    * @brief dtChangeMin: Specifies the percentage by which the maximum timescale
    * increases
    * when the timescale reaches the maximum.
    */
   virtual void ioParam_growthFactor(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   AdaptiveTimeScaleProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~AdaptiveTimeScaleProbe();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  protected:
   AdaptiveTimeScaleProbe();
   virtual void createComponents(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeOutputter(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeTrigger(char const *name, PVParams *params);
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual Response::Status prepareCheckpointWrite(double simTime) override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   Response::Status respondAdaptTimestep(std::shared_ptr<AdaptTimestepMessage const> message);
   virtual void calcValues(double timestamp) override;
   virtual void allocateTimeScaleController();

  protected:
   // Probe components, set by createComponents(), called by initialize()
   std::shared_ptr<AdaptiveTimeScaleProbeOutputter> mProbeOutputter;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger;

   ProbeDataBuffer<TimeScaleData> mStoredValues;

   char *mTargetName              = nullptr;
   double mBaseMax                = 1.0;
   double mBaseMin                = 1.0;
   double tauFactor               = 1.0;
   double mGrowthFactor           = 1.0;

   ProbeInterface *mTargetProbe                              = nullptr;
   AdaptiveTimeScaleController *mAdaptiveTimeScaleController = nullptr;

   double mBaseDeltaTime = 1.0; // The parent's DeltaTime, set during InitializeState.
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALEPROBE_HPP_ */
