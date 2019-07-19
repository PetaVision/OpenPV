/*
 * WeightedOccludingInternalStateBuffer.hpp
 *
 *  Created on: Jul 19, 2019
 *      Author: Jacob Springer
 */

#ifndef WEIGHTEDOCCLUDINGINTERNALSTATEBUFFER_HPP__
#define WEIGHTEDOCCLUDINGINTERNALSTATEBUFFER_HPP__

#include "components/InternalStateBuffer.hpp"

#include "components/OccludingGSynAccumulator.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

class WeightedOccludingInternalStateBuffer : public InternalStateBuffer {
  protected:
   virtual void ioParam_occludingLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_occlusionDepth(enum ParamsIOFlag ioFlag);

  public:
   WeightedOccludingInternalStateBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~WeightedOccludingInternalStateBuffer();

  protected:
   WeightedOccludingInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Computes the buffer as excitatory input minus inhibitory input from the LayerInput buffer.
    * The previous internal state has no effect on the new internal state.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   GSynAccumulator *mAccumulatedGSyn = nullptr;
   OccludingGSynAccumulator *mOccludingGSyn = nullptr;
   char* mOccludingLayerName = nullptr;
   int mOcclusionDepth = 0;
};

} // namespace PV

#endif // WEIGHTEDOCCLUDINGINTERNALSTATEBUFFER_HPP__
