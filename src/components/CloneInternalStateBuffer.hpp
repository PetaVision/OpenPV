/*
 * CloneInternalStateBuffer.hpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#ifndef CLONEINTERNALSTATEBUFFER_HPP_
#define CLONEINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * An InternalStateBuffer subclass that shares the data of
 * the InternalStateBuffer of another class. The ReadWritePointer
 * is null, and the BufferData pointer is the same as that of
 * the original layer.
 */
class CloneInternalStateBuffer : public InternalStateBuffer {
  protected:
   /**
    * List of parameters needed from the CloneInternalStateBuffer class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief initVType: CloneInternalStateBuffer does not use InitVType.
    */
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   CloneInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~CloneInternalStateBuffer();

  protected:
   CloneInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Sets the read-only pointer to the original layer's read-only pointer.
    */
   virtual void setReadOnlyPointer();

  protected:
   InternalStateBuffer *mOriginalBuffer = nullptr;
};

} // namespace PV

#endif // CLONEINTERNALSTATEBUFFER_HPP_
