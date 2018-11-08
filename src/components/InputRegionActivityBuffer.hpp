/*
 * InputRegionActivityBuffer.hpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#ifndef INPUTREGIONACTIVITYBUFFER_HPP_
#define INPUTREGIONACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"
#include "components/InputActivityBuffer.hpp"

namespace PV {

/**
 * A component to contain the activity buffer of a HyPerLayer.
 */
class InputRegionActivityBuffer : public ActivityBuffer {

  public:
   InputRegionActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~InputRegionActivityBuffer();

  protected:
   InputRegionActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual void setReadOnlyPointer() override;

  protected:
   InputActivityBuffer *mOriginalInput = nullptr;
};

} // namespace PV

#endif // INPUTREGIONACTIVITYBUFFER_HPP_
