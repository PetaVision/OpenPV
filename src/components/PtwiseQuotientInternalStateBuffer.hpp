/*
 * PtwiseQuotientInternalStateBuffer.hpp
 */

#ifndef PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
#define PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to compute the internal state (V) buffer as the pointwise product of the excitatory
 * and inhibitory channels of a LayerInput (GSyn) buffer.
 */
class PtwiseQuotientInternalStateBuffer : public InternalStateBuffer {
  public:
   PtwiseQuotientInternalStateBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PtwiseQuotientInternalStateBuffer();

  protected:
   PtwiseQuotientInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   LayerInputBuffer *mGSyn = nullptr;
};

} // namespace PV

#endif // PTWISEPRODUCTINTERNALSTATEBUFFER_HPP_
