/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTLAYERPROBE_HPP_
#define TRIGGERTESTLAYERPROBE_HPP_ 
#include <io/LayerProbe.hpp>

namespace PV{

class TriggerTestLayerProbe : public PV::LayerProbe{
public:
   TriggerTestLayerProbe(const char * name, HyPerCol * hc);
   virtual int outputStateWrapper(double time, double dt);
   virtual int outputState(double time);

protected:
   /**
    * @brief textOutputFlag: TriggerTestLayerProbe does not use textOutputFlag;
    * as it overrides outputStateWrapper to always create a text file.
    */
   virtual void ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) {}

   /**
    * TriggerTestLayerProbe::needRecalc(double) always returns true so that we can always
    * investigate the value of needUpdate()
    */
   virtual bool needRecalc(double timevalue) { return true; }
   
   /**
    * Sets calcValue to the value of needUpdate(timevalue, dt), where dt is the parent HyPerCol's dt.
    */
   virtual int calcValues(double timevalue);
}; // end TriggerTestLayer

BaseObject * createTriggerTestLayerProbe(char const * name, HyPerCol * hc);

}  // end namespacePV
#endif /* IMAGETESTPROBE_HPP */
