#ifndef SUMPOOLTESTINPUTLAYER_HPP_
#define SUMPOOLTESTINPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class SumPoolTestInputLayer : public PV::ANNLayer {
  public:
   SumPoolTestInputLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
}; // end class SumPoolTestInputLayer

} /* namespace PV */
#endif
