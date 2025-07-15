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
   RescaleLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~RescaleLayer();

  protected:
   RescaleLayer();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
}; // class RescaleLayer

} // namespace PV

#endif /* RESCALELAYER_HPP_ */
