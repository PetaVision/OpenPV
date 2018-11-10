/*
 * RescaleLayer.cpp
 */

#ifndef RESCALELAYER_HPP_
#define RESCALELAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

/**
 * Rescale uses the activity of a different layer and rescales it according to one of
 * several methods.
 */
class RescaleLayer : public CloneVLayer {
   // Derived from CloneVLayer for OriginalLayerNameParam and the lack of LayerInput,
   // but its ActivityComponent will not have an InternalStateBuffer.
  public:
   RescaleLayer(const char *name, HyPerCol *hc);
   virtual ~RescaleLayer();

  protected:
   RescaleLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual ActivityComponent *createActivityComponent() override;
}; // class RescaleLayer

} // namespace PV

#endif /* RESCALELAYER_HPP_ */
