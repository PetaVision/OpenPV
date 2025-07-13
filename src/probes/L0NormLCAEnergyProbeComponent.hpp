#ifndef L0NORMLCAENERGYPROBECOMPONENT_HPP_
#define L0NORMLCAENERGYPROBECOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/EnergyProbeComponent.hpp"
#include <memory>
#include <string>

namespace PV {

class L0NormLCAEnergyProbeComponent : public EnergyProbeComponent {
  protected:
   /**
    * List of parameters for the L0NormLCAEnergyProbeComponent class
    * @name L0NormLCAEnergyProbeComponent Parameters
    * @{
    */

   /**
    * @brief coefficient: L0NormLCAEnergyProbeComponent does not read the coefficient
    * parameter. Instead, it reads VThresh from a layer passed into it in initializeState()
    * and sets coefficient to that value.
    */
   virtual void ioParam_coefficient(ParamsIOSwitch ioSwitch) override;
   /** @} */

  public:
   L0NormLCAEnergyProbeComponent(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~L0NormLCAEnergyProbeComponent() {}

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   virtual void initializeState(HyPerLayer *targetLayer) override;

  protected:
   L0NormLCAEnergyProbeComponent() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
}; // class EnergyProbeComponent

} // namespace PV

#endif // L0NORMLCAENERGYPROBECOMPONENT_HPP_
