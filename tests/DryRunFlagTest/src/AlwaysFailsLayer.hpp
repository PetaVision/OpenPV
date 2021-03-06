#ifndef ALWAYSFAILSLAYER_HPP_
#define ALWAYSFAILSLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

/**
 * A layer whose needUpdate method always exits with an error.
 * The purpose is to ensure that running a params file with
 * the dry run option (-n) never executes any timesteps.
 */
class AlwaysFailsLayer : public HyPerLayer {
  public:
   AlwaysFailsLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~AlwaysFailsLayer();

  protected:
   AlwaysFailsLayer();
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual Response::Status checkUpdateState(double simTime, double deltaTime) override;
};

} // end namespace PV

#endif // ALWAYSFAILSLAYER_HPP_
