#ifndef TARGETLAYERCOMPONENT_HPP_
#define TARGETLAYERCOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeComponent.hpp"
#include <memory>
#include <string>

namespace PV {

class TargetLayerComponent : public ProbeComponent {
  protected:
   virtual void ioParam_targetLayer(ParamsIOSwitch ioSwitch);

  public:
   TargetLayerComponent(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~TargetLayerComponent();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   void ioParamsFillGroup(ParamsIOSwitch ioSwitch);

   HyPerLayer *getTargetLayer() { return mTargetLayer; }
   HyPerLayer const *getTargetLayer() const { return mTargetLayer; }
   std::string const &getTargetLayerName() const { return mTargetLayerName; }

  protected:
   TargetLayerComponent() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

  private:
   std::string mTargetLayerName;
   HyPerLayer *mTargetLayer     = nullptr;

}; // class TargetLayerComponent

} // namespace PV

#endif // TARGETLAYERCOMPONENT_HPP_
