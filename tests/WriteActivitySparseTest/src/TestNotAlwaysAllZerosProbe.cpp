/*
 * TestNotAlwaysAllZerosProbe.cpp
 *
 * A subclass of StatsProbe that verifies a layer takes a nonzero value at some point in time.
 * Once the target layer gets a nonzero value, it sets a member variable to true.
 * The public member function nonzeroValueHasOccurred() tells whether this has happened.
 * Typical use is to check this value after the run has completed but before the
 * HyPerCol is deleted.  It is useful for checking that a test isn't passing because
 * two layern
 *
 *  Created on: Apr 2, 2015
 *      Author: pschultz
 */

#include "TestNotAlwaysAllZerosProbe.hpp"

namespace PV {

TestNotAlwaysAllZerosProbe::TestNotAlwaysAllZerosProbe(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

Response::Status TestNotAlwaysAllZerosProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      Fatal().printf(
            "!!Time %f: TestNotAlwaysAllZerosProbe::outputState failed for %s\n",
            simTime,
            getTargetLayer()->getDescription_c());
   }
   for (int b = 0; b < mLocalBatchWidth; b++) {
      if (nnz[b] != 0) {
         nonzeroValueOccurred = true;
      }
   }
   return status;
}

int TestNotAlwaysAllZerosProbe::initialize_base() {
   nonzeroValueOccurred = false;
   return PV_SUCCESS;
}

void TestNotAlwaysAllZerosProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void TestNotAlwaysAllZerosProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

}; // namespace PV
