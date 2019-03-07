#ifndef COMPARISONLAYER_HPP_
#define COMPARISONLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ComparisonLayer : public PV::ANNLayer {
  public:
   ComparisonLayer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   Response::Status checkUpdateState(double timef, double dt) override;
};

} /* namespace PV */
#endif
