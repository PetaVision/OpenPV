#ifndef PROBETRIGGERCOMPONENT_HPP_
#define PROBETRIGGERCOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "components/LayerUpdateController.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeComponent.hpp"
#include <memory>

namespace PV {

class ProbeTriggerComponent : public ProbeComponent {
  protected:
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);

  public:
   ProbeTriggerComponent(char const *objName, PVParams *params);
   virtual ~ProbeTriggerComponent();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual bool needUpdate(double simTime, double deltaTime);

   char const *getTriggerLayerName() const { return mTriggerLayerName; }
   double getTriggerOffset() const { return mTriggerOffset; }

  protected:
   ProbeTriggerComponent() {}
   void initialize(char const *objName, PVParams *params);

  private:
   LayerUpdateController *mTriggerControl = nullptr;
   bool mTriggerLayerFlag                 = false;
   char *mTriggerLayerName                = nullptr;
   double mTriggerOffset                  = 0.0;
};

} // namespace PV

#endif // PROBETRIGGERCOMPONENT_HPP_
