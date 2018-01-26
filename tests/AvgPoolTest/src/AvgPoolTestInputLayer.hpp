#ifndef AVGPOOLTESTINPUTLAYER_HPP_
#define AVGPOOLTESTINPUTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class AvgPoolTestInputLayer : public PV::HyPerLayer {
  public:
   AvgPoolTestInputLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
}; // end class AvgPoolTestInputLayer

} /* namespace PV */
#endif
