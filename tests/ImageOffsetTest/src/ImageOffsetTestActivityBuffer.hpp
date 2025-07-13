#ifndef IMAGEOFFSETTESTACTIVITYBUFFER_HPP_
#define IMAGEOFFSETTESTACTIVITYBUFFER_HPP_

#include <components/ImageActivityBuffer.hpp>

namespace PV {

class ImageOffsetTestActivityBuffer : public ImageActivityBuffer {
  public:
   ImageOffsetTestActivityBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTACTIVITYBUFFER_HPP_
