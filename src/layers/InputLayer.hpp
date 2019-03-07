// InputLayer
// Base class for layers that take their input from file IO

#ifndef INPUTLAYER_HPP__
#define INPUTLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class InputLayer : public HyPerLayer {
  public:
   InputLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~InputLayer();

  protected:
   InputLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerUpdateController *createLayerUpdateController() override;

   virtual LayerInputBuffer *createLayerInput() override;

   /**
    * Each InputLayer-derived class typically has a corresponding ActivityBuffer subclass.
    * Derived classes should have a corresponding ActivityComponent subclass that contains
    * that ActivityBuffer subclass.
    */
   virtual ActivityComponent *createActivityComponent() override = 0;
};

} // end namespace PV

#endif
