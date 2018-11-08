/*
 * HyPerActivityBuffer.hpp
 *
 *  Created on: Oct 12, 2018 from code from the original HyPerLayer
 *      Author: Pete Schultz
 */

#ifndef HYPERACTIVITYBUFFER_HPP_
#define HYPERACTIVITYBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/VInputActivityBuffer.hpp"

namespace PV {

/**
 * A component to contain the activity buffer of a HyPerLayer.
 */
class HyPerActivityBuffer : public VInputActivityBuffer {

  public:
   HyPerActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~HyPerActivityBuffer();

  protected:
   HyPerActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Copies the internal state buffer to the activity buffer, taking into account the halo.
    * Note that it does not check whether the internal state buffer has updated; the calling
    * routine needs make sure to do so before updating the activity buffer.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime);
};

} // namespace PV

#endif // HYPERACTIVITYBUFFER_HPP_
