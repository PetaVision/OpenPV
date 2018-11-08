/*
 * ActivityBuffer.hpp
 *
 *  Created on: Sep 12, 2018
 *      Author: Pete Schultz
 */

#ifndef ACTIVITYBUFFER_HPP_
#define ACTIVITYBUFFER_HPP_

#include "components/ComponentBuffer.hpp"

namespace PV {

/**
 * A component to contain the activity buffer of a HyPerLayer.
 * It sets the label to "A" and the extended flag to true, but has no
 * update method defined.
 */
class ActivityBuffer : public ComponentBuffer {

  public:
   ActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~ActivityBuffer();

  protected:
   ActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // ACTIVITYBUFFER_HPP_
