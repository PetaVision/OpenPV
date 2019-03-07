#ifndef SUMPOOLTESTINPUTBUFFER_HPP_
#define SUMPOOLTESTINPUTBUFFER_HPP_

#include <components/ActivityBuffer.hpp>

namespace PV {

class SumPoolTestInputBuffer : public ActivityBuffer {
  public:
   SumPoolTestInputBuffer(const char *name, PVParams *params, Communicator const *comm);

   void updateBufferCPU(double simTime, double deltaTime) override;

}; // end class SumPoolTestInputBuffer

} /* namespace PV */
#endif
