#ifndef DROPOUTLAYER_HPP__
#define DROPOUTLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * DropoutLayer is a layer type that uses DropoutActivityBuffer.
 */
class DropoutLayer : public HyPerLayer {
  public:
   DropoutLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~DropoutLayer();

  protected:
   DropoutLayer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
