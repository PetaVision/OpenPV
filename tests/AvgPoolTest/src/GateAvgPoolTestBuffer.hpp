#ifndef GATEAVGPOOLTESTBUFFER_HPP_
#define GATEAVGPOOLTESTBUFFER_HPP_

#include <components/HyPerActivityBuffer.hpp>

namespace PV {

class GateAvgPoolTestBuffer : public HyPerActivityBuffer {
  public:
   GateAvgPoolTestBuffer(const char *name, HyPerCol *hc);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;

  private:
}; // end class GateAvgPoolTestBuffer

} /* namespace PV */
#endif // GATEAVGPOOLTESTBUFFER_HPP_
