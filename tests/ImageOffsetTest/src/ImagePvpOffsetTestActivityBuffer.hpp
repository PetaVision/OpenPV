#ifndef IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_
#define IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_

#include <components/PvpActivityBuffer.hpp>

namespace PV {

class ImagePvpOffsetTestActivityBuffer : public PvpActivityBuffer {
  protected:
   virtual void ioParam_displayPeriod(ParamsIOSwitch ioSwitch) override;

  public:
   ImagePvpOffsetTestActivityBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};
}

#endif // IMAGEPVPOFFSETTESTACTIVITYBUFFER_HPP_
