/*
 * VInputActivityBuffer.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef VINPUTACTIVITYBUFFER_HPP_
#define VINPUTACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"
#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A component to define an activity buffer with a pointer to an InternalStateBuffer (V).
 * It does not override initializeState or updateBuffer.
 * ActivityBuffer classes whose initialize and update methods depend on an InternalStateBuffer
 * can derive from this class to get an InternalStateBuffer automatically.
 */
class VInputActivityBuffer : public ActivityBuffer {

  public:
   VInputActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~VInputActivityBuffer();

  protected:
   VInputActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   InternalStateBuffer *mInternalState = nullptr;
};

} // namespace PV

#endif // VINPUTACTIVITYBUFFER_HPP_
