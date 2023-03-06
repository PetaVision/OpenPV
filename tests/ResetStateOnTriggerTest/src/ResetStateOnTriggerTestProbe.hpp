#ifndef RESETSTATEONTRIGGERTESTPROBE_HPP_
#define RESETSTATEONTRIGGERTESTPROBE_HPP_

#include "ResetStateOnTriggerTestProbeLocal.hpp"
#include "ResetStateOnTriggerTestProbeOutputter.hpp"
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <include/PVLayerLoc.h>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <observerpattern/Response.hpp>
#include <probes/TargetLayerComponent.hpp>

#include <memory>

using PV::BaseObject;
using PV::CommunicateInitInfoMessage;
using PV::Communicator;
using PV::InitializeStateMessage;
using PV::LayerOutputStateMessage;
using PV::ProbeWriteParamsMessage;
using PV::PVParams;
using PV::TargetLayerComponent;

class ResetStateOnTriggerTestProbe : public PV::BaseObject {
  public:
   ResetStateOnTriggerTestProbe(
         char const *name,
         PV::PVParams *params,
         PV::Communicator const *comm);
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

   virtual PV::Response::Status
   communicateInitInfo(std::shared_ptr<PV::CommunicateInitInfoMessage const> message) override;

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual PV::Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   PV::Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   PV::Response::Status
   respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   PV::Response::Status
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
createResetStateOnTriggerTestProbe(char const *name, PVParams *params, Communicator const *comm);

#endif // RESETSTATEONTRIGGERTESTPROBE_HPP_
