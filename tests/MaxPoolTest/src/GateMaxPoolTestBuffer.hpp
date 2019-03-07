#ifndef GATEMAXPOOLTESTBUFFER_HPP_
#define GATEMAXPOOLTESTBUFFER_HPP_

#include <components/GSynAccumulator.hpp>

namespace PV {

class GateMaxPoolTestBuffer : public GSynAccumulator {
  public:
   GateMaxPoolTestBuffer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif // GATEMAXPOOLTESTBUFFER_HPP_
