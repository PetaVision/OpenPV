#ifndef MASKTESTINPUTLAYER_HPP_
#define MASKTESTINPUTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class MaskTestInputLayer : public PV::HyPerLayer {
  public:
   MaskTestInputLayer(const char *name, HyPerCol *hc);

  protected:
   virtual ActivityComponent *createActivityComponent() override;
   virtual Response::Status updateState(double timef, double dt) override;

  private:
};

} /* namespace PV */
#endif
