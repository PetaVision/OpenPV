#ifndef RESETSTATEONTRIGGERTESTPROBE_HPP_
#define RESETSTATEONTRIGGERTESTPROBE_HPP_

#include "ResetStateOnTriggerTestProbeLocal.hpp"
#include "ResetStateOnTriggerTestProbeOutputter.hpp"
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <structures/PVLayerLoc.hpp>
#include <include/pv_common.h>
#include <params/PVParams.hpp>
#include <observerpattern/Response.hpp>
#include <probes/TargetLayerComponent.hpp>

#include <memory>

using namespace PV;

class ResetStateOnTriggerTestProbe : public BaseObject {
  public:
   ResetStateOnTriggerTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~ResetStateOnTriggerTestProbe();

   /**
    * Returns zero if the test has passed so far; returns nonzero otherwise.
    */
   bool foundDiscrepancies() const { return mProbeOutputter->foundDiscrepancies(); }

   /**
    * Returns the time of the first failure if the test has failed (i.e. getProbeStatus() returns
    * nonzero)
    * Undefined if the test is still passing.
    */
   double getFirstFailureTime() const { return mProbeOutputter->getFirstFailureTime(); }

  protected:
   ResetStateOnTriggerTestProbe() {}

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status
   respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status
   respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

   // Data members
  protected:
   double mFirstFailureTime                                               = 0.0;
   int mProbeStatus                                                       = PV_SUCCESS;
   std::shared_ptr<ResetStateOnTriggerTestProbeLocal> mProbeLocal         = nullptr;
   std::shared_ptr<ResetStateOnTriggerTestProbeOutputter> mProbeOutputter = nullptr;
   float const *mTargetLayerData                                          = nullptr;
   PVLayerLoc const *mTargetLayerLoc                                      = nullptr;
   std::shared_ptr<TargetLayerComponent> mTargetLayerLocator              = nullptr;
};

BaseObject *
createResetStateOnTriggerTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

#endif // RESETSTATEONTRIGGERTESTPROBE_HPP_
