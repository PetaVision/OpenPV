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
   TriggerTestLayerProbe(const char *name, HyPerCol *hc);
   virtual Response::Status outputStateWrapper(double time, double dt) override;
   virtual Response::Status outputState(double timestamp) override;

  protected:
   /**
    * @brief textOutputFlag: TriggerTestLayerProbe does not use textOutputFlag;
    * as it overrides outputStateWrapper to always create a text file.
    */
   virtual void ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) override {}

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
}; // end TriggerTestLayer

} // end namespacePV
#endif /* IMAGETESTPROBE_HPP */
