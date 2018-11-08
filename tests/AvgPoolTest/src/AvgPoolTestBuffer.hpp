#ifndef AVGPOOLTESTBUFFER_HPP_
#define AVGPOOLTESTBUFFER_HPP_

#include <components/HyPerActivityBuffer.hpp>

namespace PV {

class AvgPoolTestBuffer : public HyPerActivityBuffer {
  public:
   AvgPoolTestBuffer(const char *name, PVParams *params, Communicator *comm);

  protected:
   void updateBufferCPU(double simTime, double deltaTime) override;

}; // end class AvgPoolTestBuffer

} /* namespace PV */
#endif
