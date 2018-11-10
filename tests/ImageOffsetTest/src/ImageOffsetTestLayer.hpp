#ifndef IMAGEOFFSETTESTLAYER_HPP_
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/ImageLayer.hpp>

namespace PV {
class ImageOffsetTestLayer : public ImageLayer {
  public:
   ImageOffsetTestLayer(char const *name, HyPerCol *hc);
   virtual ~ImageOffsetTestLayer();

  protected:
   ImageOffsetTestLayer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;

   virtual void setNontriggerDeltaUpdateTime(double dt) override;
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTLAYER_HPP_
