#ifndef IMAGECOLLATIONLAYER_HPP__
#define IMAGECOLLATIONLAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class ImageCollationLayer : public InputLayer {

  public:
   ImageCollationLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~ImageCollationLayer();

  protected:
   ImageCollationLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // IMAGECOLLATIONLAYER_HPP__
