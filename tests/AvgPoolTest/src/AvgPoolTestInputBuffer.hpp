#ifndef AVGPOOLTESTINPUTBUFFER_HPP_
#define AVGPOOLTESTINPUTBUFFER_HPP_

#include <components/ActivityBuffer.hpp>

namespace PV {

class AvgPoolTestInputBuffer : public ActivityBuffer {
  public:
   AvgPoolTestInputBuffer(const char *name, HyPerCol *hc);

  protected:
   void updateBufferCPU(double timef, double dt) override;

  private:
}; // end class AvgPoolTestInputBuffer

} /* namespace PV */
#endif
