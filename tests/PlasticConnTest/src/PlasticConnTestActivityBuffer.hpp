/*
 * PlasticConnTestActivityBuffer.hpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#ifndef PLASTICCONNTESTACTIVITYBUFFER_HPP_
#define PLASTICCONNTESTACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

namespace PV {

/**
 * PlasticConnTestActivityBuffer is the ActivityBuffer subclass for MPITestLayer
 */
class PlasticConnTestActivityBuffer : public ActivityBuffer {
  public:
   PlasticConnTestActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PlasticConnTestActivityBuffer();

  protected:
   PlasticConnTestActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   void setActivityToGlobalPos();
};

} // namespace PV

#endif // PLASTICCONNTESTACTIVITYBUFFER_HPP_
