#ifndef GATEMAXPOOLTESTLAYER_HPP_
#define GATEMAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class GateMaxPoolTestLayer : public PV::ANNLayer {
  public:
   GateMaxPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
};

} /* namespace PV */
#endif // GATEMAXPOOLTESTLAYER_HPP_
