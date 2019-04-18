#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpOffsetTestLayer : public PV::PvpLayer {
  public:
   ImagePvpOffsetTestLayer(const char *name, HyPerCol *hc);
   virtual double getDeltaUpdateTime() override;

  protected:
   Response::Status updateState(double timef, double dt) override;
   bool readyForNextFile() override;
};
}

#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
