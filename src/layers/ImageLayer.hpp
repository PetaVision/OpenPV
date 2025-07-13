#ifndef IMAGELAYER_HPP__
#define IMAGELAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class ImageLayer : public InputLayer {

  public:
   ImageLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~ImageLayer();

  protected:
   ImageLayer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // IMAGELAYER_HPP__
