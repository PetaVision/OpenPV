#ifndef IMAGELAYER_HPP__
#define IMAGELAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class ImageLayer : public InputLayer {

  public:
   ImageLayer(char const *name, HyPerCol *hc);
   virtual ~ImageLayer();

  protected:
   ImageLayer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // IMAGELAYER_HPP__
