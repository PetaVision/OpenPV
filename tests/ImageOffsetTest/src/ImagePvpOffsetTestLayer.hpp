#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpOffsetTestLayer : public PvpLayer {
  public:
   ImagePvpOffsetTestLayer(char const *name, PVParams *params, Communicator *comm);
   virtual ~ImagePvpOffsetTestLayer();

  protected:
   ImagePvpOffsetTestLayer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;

   virtual void setNontriggerDeltaUpdateTime(double dt) override;
};
}

#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
