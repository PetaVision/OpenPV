/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTLAYER_HPP_
#define TRIGGERTESTLAYER_HPP_ 
#include <layers/HyPerLayer.hpp>

namespace PV{

class TriggerTestLayer : public PV::HyPerLayer{
public:
   TriggerTestLayer(const char * name, HyPerCol * hc);
   virtual bool activityIsSpiking() { return false; }
   int virtual updateStateWrapper (double time, double dt);
};

BaseObject * createTriggerTestLayer(char const * name, HyPerCol * hc);

}
#endif /* IMAGETESTPROBE_HPP */
