/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTLAYERPROBE_HPP_
#define TRIGGERTESTLAYERPROBE_HPP_
#include "probes/LayerProbe.hpp"

namespace PV {

class TriggerTestLayerProbe : public PV::LayerProbe {
  public:
   TriggerTestLayerProbe(const char *name, PVParams *params, Communicator *comm);
   virtual Response::Status outputStateWrapper(double simTime, double dt) override;

  protected:
   /**
    * @brief textOutputFlag: TriggerTestLayerProbe does not use textOutputFlag;
    * as it overrides outputStateWrapper to always create a text file.
    */
   virtual void ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) override {}

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * TriggerTestLayerProbe::needRecalc(double) always returns true so that we can always
    * investigate the value of needUpdate()
    */
   virtual bool needRecalc(double timevalue) override { return true; }

   /**
    * Sets calcValue to the value of needUpdate(timevalue, dt), where dt is the parent HyPerCol's
    * dt.
    */
   virtual void calcValues(double timevalue) override;

   /**
    * TriggerTestLayerProbe does not call outputState, but the routine is needed since
    * LayerProbe::TriggerTestLayerProbe is pure virtual
    */
   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   double mDeltaTime = 1.0; // Set during InitializeState, and used in calcValues.
}; // end TriggerTestLayer

} // end namespacePV
#endif /* IMAGETESTPROBE_HPP */
