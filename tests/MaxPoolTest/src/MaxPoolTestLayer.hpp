#ifndef MAXPOOLTESTLAYER_HPP_
#define MAXPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class MaxPoolTestLayer : public HyPerLayer {
  public:
   MaxPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   ActivityComponent *createActivityComponent() override;
};

} /* namespace PV */
#endif
