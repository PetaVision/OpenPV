#ifndef BINNINGTESTLAYER_HPP_
#define BINNINGTESTLAYER_HPP_

#include <layers/BinningLayer.hpp>

namespace PV {

class BinningTestLayer : public PV::BinningLayer {
  public:
   BinningTestLayer(const char *name, HyPerCol *hc);

  protected:
   Response::Status updateState(double timef, double dt) override;

  private:
};

} /* namespace PV */
#endif
