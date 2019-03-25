#ifndef IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_
#define IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_

#include <components/PvpActivityBuffer.hpp>

namespace PV {

class ImagePvpOffsetTestActivityBuffer : public PvpActivityBuffer {
  protected:
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag) override;

  public:
   ImagePvpOffsetTestActivityBuffer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};
}

#endif // IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_
