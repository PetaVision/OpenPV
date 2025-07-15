#ifndef MAXPOOLTESTLAYER_HPP_
#define MAXPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class MaxPoolTestLayer : public HyPerLayer {
  public:
   MaxPoolTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
};

} /* namespace PV */
#endif
