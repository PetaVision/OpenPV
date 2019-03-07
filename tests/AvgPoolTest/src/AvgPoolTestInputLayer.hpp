#ifndef AVGPOOLTESTINPUTLAYER_HPP_
#define AVGPOOLTESTINPUTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class AvgPoolTestInputLayer : public HyPerLayer {
  public:
   AvgPoolTestInputLayer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   ActivityComponent *createActivityComponent() override;
}; // end class AvgPoolTestInputLayer

} /* namespace PV */
#endif
