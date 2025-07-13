#ifndef VTHRESHENERGYPROBECOMPONENT_HPP_
#define VTHRESHENERGYPROBECOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/EnergyProbeComponent.hpp"
#include <memory>
#include <string>

namespace PV {

class VThreshEnergyProbeComponent : public EnergyProbeComponent {
  protected:
   /**
    * List of parameters for the VThreshEnergyProbeComponent class
    * @name VThreshEnergyProbeComponent Parameters
    * @{
    */

   /**
    * @brief coefficient: VThreshEnergyProbeComponent does not read the coefficient
    * parameter. Instead, it reads VThresh from a layer passed into it in initializeState()
    * and sets coefficient to that value.
    */
   virtual void ioParam_coefficient(ParamsIOSwitch ioSwitch) override;
   /** @} */

  public:
   VThreshEnergyProbeComponent(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~VThreshEnergyProbeComponent() {}

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   virtual void initializeState(HyPerLayer *targetLayer) override;

  protected:
   VThreshEnergyProbeComponent() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
}; // class EnergyProbeComponent

} // namespace PV

#endif // VTHRESHENERGYPROBECOMPONENT_HPP_
