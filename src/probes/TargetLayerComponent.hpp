#ifndef TARGETLAYERCOMPONENT_HPP_
#define TARGETLAYERCOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeComponent.hpp"
#include <memory>
#include <string>

namespace PV {

class TargetLayerComponent : public ProbeComponent {
  protected:
   virtual void ioParam_targetLayer(enum ParamsIOFlag ioFlag);

  public:
   TargetLayerComponent(char const *objName, PVParams *params);
   virtual ~TargetLayerComponent();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   HyPerLayer *getTargetLayer() { return mTargetLayer; }
   HyPerLayer const *getTargetLayer() const { return mTargetLayer; }
   char const *getTargetLayerName() const { return mTargetLayerName; }

  protected:
   TargetLayerComponent() {}
   void initialize(char const *objName, PVParams *params);

  private:
   char *mTargetLayerName   = nullptr;
   HyPerLayer *mTargetLayer = nullptr;

}; // class TargetLayerComponent

} // namespace PV

#endif // TARGETLAYERCOMPONENT_HPP_
