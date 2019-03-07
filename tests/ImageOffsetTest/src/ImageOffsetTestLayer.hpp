#ifndef IMAGEOFFSETTESTLAYER_HPP_
#define IMAGEOFFSETTESTLAYER_HPP_

#include <layers/ImageLayer.hpp>

namespace PV {
class ImageOffsetTestLayer : public ImageLayer {
  public:
   ImageOffsetTestLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~ImageOffsetTestLayer();

  protected:
   ImageOffsetTestLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTLAYER_HPP_
