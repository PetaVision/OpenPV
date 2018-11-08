#ifndef MAXPOOLTESTBUFFER_HPP_
#define MAXPOOLTESTBUFFER_HPP_

#include <components/ANNActivityBuffer.hpp>

namespace PV {

class MaxPoolTestBuffer : public ANNActivityBuffer {
  public:
   MaxPoolTestBuffer(const char *name, PVParams *params, Communicator *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif
