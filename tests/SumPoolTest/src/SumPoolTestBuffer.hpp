#ifndef SUMPOOLTESTBUFFER_HPP_
#define SUMPOOLTESTBUFFER_HPP_

#include <components/ANNActivityBuffer.hpp>

namespace PV {

class SumPoolTestBuffer : public ANNActivityBuffer {
  public:
   SumPoolTestBuffer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;

}; // end class SumPoolTestBuffer

} /* namespace PV */
#endif
