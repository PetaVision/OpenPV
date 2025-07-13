#ifndef PROBETRIGGERCOMPONENT_HPP_
#define PROBETRIGGERCOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "components/LayerUpdateController.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeComponent.hpp"
#include <memory>

namespace PV {

class ProbeTriggerComponent : public ProbeComponent {
  protected:
   virtual void ioParam_triggerLayerName(ParamsIOSwitch ioSwitch);
   virtual void ioParam_triggerOffset(ParamsIOSwitch ioSwitch);

  public:
   ProbeTriggerComponent(
           std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~ProbeTriggerComponent();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   void ioParamsFillGroup(ParamsIOSwitch ioSwitch);

   virtual bool needUpdate(double simTime, double deltaTime);

   std::string const &getTriggerLayerName() const { return mTriggerLayerName; }
   double getTriggerOffset() const { return mTriggerOffset; }

  protected:
   ProbeTriggerComponent() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

  private:
   LayerUpdateController *mTriggerControl = nullptr;
   bool mTriggerLayerFlag                 = false;
   std::string mTriggerLayerName;
   double mTriggerOffset = 0.0;
};

} // namespace PV

#endif // PROBETRIGGERCOMPONENT_HPP_
