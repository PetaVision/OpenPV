#ifndef MAXPOOLTESTBUFFER_HPP_
#define MAXPOOLTESTBUFFER_HPP_

#include <components/ANNActivityBuffer.hpp>

namespace PV {

class MaxPoolTestBuffer : public ANNActivityBuffer {
  public:
   MaxPoolTestBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif
