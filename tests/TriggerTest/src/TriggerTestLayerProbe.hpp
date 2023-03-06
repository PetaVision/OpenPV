/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTLAYERPROBE_HPP_
#define TRIGGERTESTLAYERPROBE_HPP_
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <io/PVParams.hpp>
#include <observerpattern/Response.hpp>
#include <probes/ProbeTriggerComponent.hpp>
#include <probes/TargetLayerComponent.hpp>

#include <memory>

namespace PV {

class TriggerTestLayerProbe : public PV::BaseObject {
  public:
   TriggerTestLayerProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

  protected:
   std::shared_ptr<TargetLayerComponent> mProbeTargetLayerLocator = nullptr;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger           = nullptr;

   static int const mInputDisplayPeriod = 5;

}; // end TriggerTestLayer

} // namespace PV
#endif /* IMAGETESTPROBE_HPP */
