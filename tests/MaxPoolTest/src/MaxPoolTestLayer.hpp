#ifndef MAXPOOLTESTLAYER_HPP_
#define MAXPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class MaxPoolTestLayer : public HyPerLayer {
  public:
   MaxPoolTestLayer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
};

} /* namespace PV */
#endif
