#ifndef ENERGYPROBECOMPONENT_HPP_
#define ENERGYPROBECOMPONENT_HPP_

#include "columns/Messages.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ColumnEnergyProbe.hpp"
#include "probes/ProbeComponent.hpp"
#include <memory>

namespace PV {

class EnergyProbeComponent : public ProbeComponent {
  protected:
   /**
    * List of parameters for the EnergyProbeComponent class
    * @name BaseProbe Parameters
    * @{
    */

   /**
    * @brief coefficient: If energyProbe is set, the coefficient parameter specifies that the
    * ColumnEnergyProbe object multiplies the result of this probe's values method by
    * Coefficient when computing the total energy.
    * @details Note that coefficient does not affect the values computed by the probe itself;
    * it is the ColumnEnergyProbe object does the multiplication.
    */
   virtual void ioParam_coefficient(ParamsIOSwitch ioSwitch);

   /**
    * @brief energyProbe: If nonblank, specifies the name of a ColumnEnergyProbe
    * that this probe contributes an energy term to.
    */
   virtual void ioParam_energyProbe(ParamsIOSwitch ioSwitch);
   /** @} */

  public:
   EnergyProbeComponent(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults);
   virtual ~EnergyProbeComponent();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);
   virtual void initializeState(HyPerLayer *targetLayer) {}
   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch);

   double getCoefficient() const { return mCoefficient; }
   ColumnEnergyProbe *getEnergyProbe() { return mEnergyProbe; }
   ColumnEnergyProbe const *getEnergyProbe() const { return mEnergyProbe; }

  protected:
   EnergyProbeComponent() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   void setCoefficient(double coefficient) { mCoefficient = coefficient; }

  private:
   double mCoefficient             = 1.0;
   ColumnEnergyProbe *mEnergyProbe = nullptr;
   std::string mEnergyProbeName;

}; // class EnergyProbeComponent

} // namespace PV

#endif // ENERGYPROBECOMPONENT_HPP_
