#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpOffsetTestLayer : public PvpLayer {
  public:
   ImagePvpOffsetTestLayer(char const *name, HyPerCol *hc);
   virtual ~ImagePvpOffsetTestLayer();

  protected:
   ImagePvpOffsetTestLayer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;

   virtual void setNontriggerDeltaUpdateTime(double dt) override;
};
}

#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
