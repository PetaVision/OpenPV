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
  protected:
   /**
    * IndexLayer does not use InitVType. Instead, the layer is initialized
    * so that global restricted index k is k*startTime.
    */
   void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;

  public:
   IndexLayer(char const *name, HyPerCol *hc);
   ~IndexLayer();

  protected:
   IndexLayer();
   int initialize(char const *name, HyPerCol *hc);
   virtual Response::Status initializeState() override;
   virtual Response::Status updateState(double timef, double dt) override;

  private:
   int initialize_base();
};

} // end namespace PV
