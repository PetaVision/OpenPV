#ifndef GATEMAXPOOLTESTLAYER_HPP_
#define GATEMAXPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class GateMaxPoolTestLayer : public HyPerLayer {
  public:
   GateMaxPoolTestLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
}; // end class GateMaxPoolTestLayer

} /* namespace PV */
#endif // GATEMAXPOOLTESTLAYER_HPP_
