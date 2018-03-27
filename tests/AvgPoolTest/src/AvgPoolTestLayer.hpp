#ifndef AVGPOOLTESTLAYER_HPP_
#define AVGPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class AvgPoolTestLayer : public PV::HyPerLayer {
  public:
   AvgPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
}; // end class AvgPoolTestLayer

} /* namespace PV */
#endif
