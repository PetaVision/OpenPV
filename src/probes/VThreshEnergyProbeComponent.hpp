#ifndef VTHRESHENERGYPROBECOMPONENT_HPP_
#define VTHRESHENERGYPROBECOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
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
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag) override;
   /** @} */

  public:
   VThreshEnergyProbeComponent(char const *objName, PVParams *params);
   virtual ~VThreshEnergyProbeComponent() {}

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   virtual void initializeState(HyPerLayer *targetLayer) override;

  protected:
   VThreshEnergyProbeComponent() {}
   void initialize(char const *objName, PVParams *params);
}; // class EnergyProbeComponent

} // namespace PV

#endif // VTHRESHENERGYPROBECOMPONENT_HPP_
