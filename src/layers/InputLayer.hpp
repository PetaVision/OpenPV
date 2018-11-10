// InputLayer
// Base class for layers that take their input from file IO

#ifndef INPUTLAYER_HPP__
#define INPUTLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class InputLayer : public HyPerLayer {
  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name InputLayer Parameters
    * @{
    */

   /**
    * triggerLayerName: InputLayer and derived classes do not use triggering, and always set
    * triggerLayerName to NULL.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   /** @} */

  public:
   InputLayer(const char *name, HyPerCol *hc);
   virtual ~InputLayer();

  protected:
   InputLayer() {}

   int initialize(const char *name, HyPerCol *hc);

   virtual LayerInputBuffer *createLayerInput() override;

   /**
    * Each InputLayer-derived class typically has a corresponding ActivityBuffer subclass.
    * Derived classes should have a corresponding ActivityComponent subclass that contains
    * that ActivityBuffer subclass.
    */
   virtual ActivityComponent *createActivityComponent() override = 0;

   virtual void setNontriggerDeltaUpdateTime(double dt) override;
};

} // end namespace PV

#endif
