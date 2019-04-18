#ifndef MASKTESTINPUTLAYER_HPP_
#define MASKTESTINPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaskTestInputLayer : public PV::ANNLayer {
  public:
   MaskTestInputLayer(const char *name, HyPerCol *hc);

  protected:
   virtual Response::Status updateState(double timef, double dt) override;

  private:
};

} /* namespace PV */
#endif
