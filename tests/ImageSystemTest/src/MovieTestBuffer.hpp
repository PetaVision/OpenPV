/*
 * MovieTestBuffer.hpp
 * Author: slundquist
 */

#ifndef MOVIETESTBUFFER_HPP_
#define MOVIETESTBUFFER_HPP_
#include <components/ImageActivityBuffer.hpp>

namespace PV {

class MovieTestBuffer : public PV::ImageActivityBuffer {
  public:
   MovieTestBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};
}

#endif // MOVIETESTBUFFER_HPP_
