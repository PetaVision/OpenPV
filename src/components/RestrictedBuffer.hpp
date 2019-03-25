/*
 * RestrictedBuffer.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef RESTRICTEDBUFFER_HPP_
#define RESTRICTEDBUFFER_HPP_

#include "components/ComponentBuffer.hpp"

namespace PV {

/**
 * A ComponentBuffer for a restricted buffer. The buffer label can be set using the
 * setBufferLabel() method. This should be done before the RegisterData phase to
 * enable checkpointing.
 */
class RestrictedBuffer : public ComponentBuffer {
  public:
   RestrictedBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~RestrictedBuffer();

  protected:
   RestrictedBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // RESTRICTEDBUFFER_HPP_
