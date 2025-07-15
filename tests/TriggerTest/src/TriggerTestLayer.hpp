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
   TriggerTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual Response::Status checkUpdateState(double simTime, double deltaTime) override;
};
}
#endif /* IMAGETESTPROBE_HPP */
