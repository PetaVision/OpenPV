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
   IndexLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~IndexLayer();

  protected:
   IndexLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV
