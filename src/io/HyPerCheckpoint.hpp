/*
 * HyPerCheckpoint.hpp
 *
 *  Created on: Oct 2, 2016
 *      Author: Pete Schultz
 *
 *  This class is a temporary convenience for the refactoring of checkpointing into the Checkpointer
 *  class. Checkpointer::checkpointRead/Write methods get a HyPerCheckpoint and they call that
 *  object's checkpointRead/Write methods. HyPerCol derives from HyPerCheckpoint. When all
 *  checkpointing is done by calling Checkpointer::registerCheckpointData so that nothing is done by
 *  checkpointRead and checkpointWrite methods except for those of the Checkpointer object, the
 *  checkpointer will no longer need to take a HyPerCheckpoint argument and this class wil
 * disappear.
 */

#ifndef HYPERCHECKPOINT_HPP_
#define HYPERCHECKPOINT_HPP_

#include "include/pv_common.h"

namespace PV {

class HyPerCheckpoint {
  public:
   HyPerCheckpoint() {}
   virtual ~HyPerCheckpoint() {}
   virtual int checkpointWrite(const char *cpDir) { return PV_SUCCESS; }
};
}

#endif // HYPERCHECKPOINT_HPP_
