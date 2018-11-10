#ifndef GATESUMPOOLTESTBUFFER_HPP_
#define GATESUMPOOLTESTBUFFER_HPP_

#include <components/ANNActivityBuffer.hpp>

namespace PV {

class GateSumPoolTestBuffer : public ANNActivityBuffer {
  public:
   GateSumPoolTestBuffer(const char *name, HyPerCol *hc);

   void updateBufferCPU(double simTime, double deltaTime) override;
}; // end class GateSumPoolTestBuffer

} /* namespace PV */
#endif // GATESUMPOOLTESTBUFFER_HPP_
