/*
 * GapActivityBuffer.hpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef GAPACTIVITYBUFFER_HPP_
#define GAPACTIVITYBUFFER_HPP_

#include "HyPerActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"

namespace PV {

/**
 * GapActivityBuffer can be used to implement gap junctions
 */
class GapActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the GapActivityBuffer class
    * @name GapLayer Parameters
    * @{
    */

   /**
    * ampSpikelet: Whereever the original activity buffer is active, the GapActivityBuffer
    * adds a spikelet of this amplitude. Default = 50.
    */
   virtual void ioParam_ampSpikelet(enum ParamsIOFlag ioFlag);

   /** @} */

  public:
   GapActivityBuffer(const char *name, PVParams *params, Communicator *comm);
   virtual ~GapActivityBuffer();

  protected:
   GapActivityBuffer();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mAmpSpikelet = 50.0f;

   ActivityBuffer *mOriginalActivity = nullptr;

}; // class GapActivityBuffer

} // namespace PV

#endif /* GAPACTIVITYBUFFER_HPP_ */
