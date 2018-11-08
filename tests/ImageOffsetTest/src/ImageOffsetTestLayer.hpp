#ifndef IMAGEOFFSETTESTLAYER_HPP_
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/ImageLayer.hpp>

namespace PV {
class ImageOffsetTestLayer : public ImageLayer {
  public:
   ImageOffsetTestLayer(char const *name, PVParams *params, Communicator *comm);
   virtual ~ImageOffsetTestLayer();

  protected:
   ImageOffsetTestLayer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;

   virtual void setNontriggerDeltaUpdateTime(double dt) override;
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTLAYER_HPP_
