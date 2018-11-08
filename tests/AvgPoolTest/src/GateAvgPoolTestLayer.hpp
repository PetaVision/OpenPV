#ifndef GATEAVGPOOLTESTLAYER_HPP_
#define GATEAVGPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class GateAvgPoolTestLayer : public HyPerLayer {
  public:
   GateAvgPoolTestLayer(const char *name, PVParams *params, Communicator *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
}; // end class GateAvgPoolTestLayer

} /* namespace PV */
#endif // GATEAVGPOOLTESTLAYER_HPP_
