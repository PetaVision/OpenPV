#ifndef AVGPOOLTESTLAYER_HPP_
#define AVGPOOLTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class AvgPoolTestLayer : public HyPerLayer {
  public:
   AvgPoolTestLayer(const char *name, PVParams *params, Communicator *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
}; // end class AvgPoolTestLayer

} /* namespace PV */
#endif
