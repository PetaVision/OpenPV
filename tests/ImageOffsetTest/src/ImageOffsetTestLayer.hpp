#ifndef IMAGEOFFSETTESTLAYER_HPP_
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/ImageLayer.hpp>

namespace PV {
class ImageOffsetTestLayer : public PV::ImageLayer {
  public:
   ImageOffsetTestLayer(const char *name, HyPerCol *hc);
   virtual double getDeltaUpdateTime();

  protected:
   int updateState(double timef, double dt);
   bool readyForNextFile();
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTLAYER_HPP_
