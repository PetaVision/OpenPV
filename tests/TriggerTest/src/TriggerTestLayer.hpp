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
   TriggerTestLayer(const char *name, PVParams *params, Communicator *comm);
   virtual bool activityIsSpiking() override { return false; }
   virtual Response::Status checkUpdateState(double simTime, double deltaTime) override;
};
}
#endif /* IMAGETESTPROBE_HPP */
