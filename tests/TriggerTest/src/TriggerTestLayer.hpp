/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTLAYER_HPP_
#define TRIGGERTESTLAYER_HPP_
#include <layers/HyPerLayer.hpp>

namespace PV {

class TriggerTestLayer : public PV::HyPerLayer {
  public:
   TriggerTestLayer(const char *name, HyPerCol *hc);
   virtual bool activityIsSpiking() override { return false; }
   Response::Status virtual updateState(double time, double dt) override;
};
}
#endif /* IMAGETESTPROBE_HPP */
