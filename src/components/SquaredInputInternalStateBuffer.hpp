/*
 * SquaredInputInternalStateBuffer.hpp
 */

#ifndef SQUAREDINPUTINTERNALSTATEBUFFER_HPP_
#define SQUAREDINPUTINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class SquaredInputInternalStateBuffer : public InternalStateBuffer {
  public:
   SquaredInputInternalStateBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~SquaredInputInternalStateBuffer();

  protected:
   SquaredInputInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Computes the buffer as the point-by-point square of the LayerInput buffer.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   LayerInputBuffer *mGSyn = nullptr;
};

} // namespace PV

#endif // SQUAREDINPUTINTERNALSTATEBUFFER_HPP_
