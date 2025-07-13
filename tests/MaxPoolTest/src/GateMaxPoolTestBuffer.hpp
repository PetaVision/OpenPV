#ifndef GATEMAXPOOLTESTBUFFER_HPP_
#define GATEMAXPOOLTESTBUFFER_HPP_

#include <components/GSynAccumulator.hpp>

namespace PV {

class GateMaxPoolTestBuffer : public GSynAccumulator {
  public:
   GateMaxPoolTestBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   void updateBufferCPU(double simTime, double deltaTime) override;
};

} /* namespace PV */
#endif // GATEMAXPOOLTESTBUFFER_HPP_
