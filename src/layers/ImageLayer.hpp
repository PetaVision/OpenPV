#ifndef IMAGELAYER_HPP__
#define IMAGELAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class ImageLayer : public InputLayer {

  public:
   ImageLayer(char const *name, PVParams *params, Communicator *comm);
   virtual ~ImageLayer();

  protected:
   ImageLayer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // IMAGELAYER_HPP__
