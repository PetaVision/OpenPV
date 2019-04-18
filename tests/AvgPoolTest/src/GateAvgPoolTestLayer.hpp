#ifndef GATEAVGPOOLTESTLAYER_HPP_
#define GATEAVGPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class GateAvgPoolTestLayer : public PV::HyPerLayer {
  public:
   GateAvgPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
}; // end class GateAvgPoolTestLayer

} /* namespace PV */
#endif // GATEAVGPOOLTESTLAYER_HPP_
