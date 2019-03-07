/*
 * MoviePvpTestBuffer.hpp
 * Author: slundquist
 */

#ifndef MOVIEPVPTESTBUFFER_HPP_
#define MOVIEPVPTESTBUFFER_HPP_
#include <components/PvpActivityBuffer.hpp>

namespace PV {

class MoviePvpTestBuffer : public PvpActivityBuffer {
  public:
   MoviePvpTestBuffer(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};

} // end namespace PV

#endif // MOVIEPVPTESTBUFFER_HPP_
