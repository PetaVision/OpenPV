#ifndef IMAGEOFFSETTESTACTIVITYBUFFER_HPP_
#define IMAGEOFFSETTESTACTIVITYBUFFER_HPP_

#include <components/ImageActivityBuffer.hpp>

namespace PV {

class ImageOffsetTestActivityBuffer : public ImageActivityBuffer {
  public:
   ImageOffsetTestActivityBuffer(const char *name, HyPerCol *hc);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */

#endif // IMAGEOFFSETTESTACTIVITYBUFFER_HPP_
