#ifndef DROPOUTLAYER_HPP__
#define DROPOUTLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * DropoutLayer is a layer type that uses DropoutActivityBuffer.
 */
class DropoutLayer : public HyPerLayer {
  public:
   DropoutLayer(const char *name, HyPerCol *hc);
   virtual ~DropoutLayer();

  protected:
   DropoutLayer() {}

   int initialize(const char *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
