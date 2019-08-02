#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/PvpLayer.hpp>

namespace PV {
class ImagePvpOffsetTestLayer : public PvpLayer {
  public:
   ImagePvpOffsetTestLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~ImagePvpOffsetTestLayer();

  protected:
   ImagePvpOffsetTestLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; /* class ImageOffsetTestLayer */

} /* namespace PV */

#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
