#ifndef GATEMAXPOOLTESTLAYER_HPP_
#define GATEMAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class GateMaxPoolTestLayer : public PV::ANNLayer {
  public:
   GateMaxPoolTestLayer(const char *name, HyPerCol *hc);

  protected:
   int updateState(double timef, double dt);

  private:
};

} /* namespace PV */
#endif // GATEMAXPOOLTESTLAYER_HPP_
