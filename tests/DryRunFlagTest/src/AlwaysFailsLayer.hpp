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
   AlwaysFailsLayer(char const *name, PVParams *params, Communicator *comm);
   virtual ~AlwaysFailsLayer();
   virtual bool needUpdate(double simTime, double dt) const override;

  protected:
   AlwaysFailsLayer();
   void initialize(char const *name, PVParams *params, Communicator *comm);

  private:
   int initialize_base();
};

} // end namespace PV

#endif // ALWAYSFAILSLAYER_HPP_
