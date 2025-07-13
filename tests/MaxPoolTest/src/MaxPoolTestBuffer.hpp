#ifndef MAXPOOLTESTBUFFER_HPP_
#define MAXPOOLTESTBUFFER_HPP_

#include <components/ANNActivityBuffer.hpp>

namespace PV {

class MaxPoolTestBuffer : public ANNActivityBuffer {
  public:
   MaxPoolTestBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif
