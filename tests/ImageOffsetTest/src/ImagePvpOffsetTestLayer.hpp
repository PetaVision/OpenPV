#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpOffsetTestLayer : public PV::PvpLayer {
  public:
   ImagePvpOffsetTestLayer(const char *name, HyPerCol *hc);
   virtual double getDeltaUpdateTime();

  protected:
   int updateState(double timef, double dt);
   bool readyForNextFile();
};
}

#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
