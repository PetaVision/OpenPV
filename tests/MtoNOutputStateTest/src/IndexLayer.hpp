/**
 * IndexLayer.hpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 *  A layer class meant to be useful in testing.
 *  At time t, the value of global batch element b, global restricted index k,
 *  is t*(b*N+k), where N is the number of neurons in global restrictes space.
 *
 */

#include <layers/HyPerLayer.hpp>

namespace PV {

class IndexLayer : public HyPerLayer {
  public:
   IndexLayer(char const *name, HyPerCol *hc);
   ~IndexLayer();

  protected:
   IndexLayer();
   int initialize(char const *name, HyPerCol *hc);
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   virtual Response::Status updateState(double timef, double dt) override;

  private:
   int initialize_base();
};

} // end namespace PV
