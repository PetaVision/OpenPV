#ifndef GATESUMPOOLTESTLAYER_HPP_
#define GATESUMPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class GateSumPoolTestLayer : public HyPerLayer {
  public:
   GateSumPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   ActivityComponent *createActivityComponent() override;
}; // end class GateSumPoolTestLayer

} /* namespace PV */
#endif // GATESUMPOOLTESTLAYER_HPP_
