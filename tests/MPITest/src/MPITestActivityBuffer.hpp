/*
 * MPITestActivityBuffer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef MPITESTACTIVITYBUFFER_HPP_
#define MPITESTACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

namespace PV {

/**
 * MPITestActivityBuffer is the ActivityBuffer subclass for MPITestLayer
 */
class MPITestActivityBuffer : public ActivityBuffer {
  public:
   MPITestActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~MPITestActivityBuffer();

  protected:
   MPITestActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   void setActivityToGlobalPos();
};

} // namespace PV

#endif // MPITESTACTIVITYBUFFER_HPP_
