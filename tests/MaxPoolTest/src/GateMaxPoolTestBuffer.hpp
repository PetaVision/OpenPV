#ifndef GATEMAXPOOLTESTBUFFER_HPP_
#define GATEMAXPOOLTESTBUFFER_HPP_

#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

class GateMaxPoolTestBuffer : public HyPerInternalStateBuffer {
  public:
   GateMaxPoolTestBuffer(const char *name, PVParams *params, Communicator *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif // GATEMAXPOOLTESTBUFFER_HPP_
