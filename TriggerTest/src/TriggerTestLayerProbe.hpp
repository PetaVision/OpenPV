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
};

}
#endif /* IMAGETESTPROBE_HPP */
