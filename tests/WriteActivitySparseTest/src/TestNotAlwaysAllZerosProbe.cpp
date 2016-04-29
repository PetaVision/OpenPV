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

TestNotAlwaysAllZerosProbe::TestNotAlwaysAllZerosProbe(const char * probeName, HyPerCol * hc) {
   initTestNotAlwaysAllZerosProbe_base();
   initTestNotAlwaysAllZerosProbe(probeName, hc);
}

int TestNotAlwaysAllZerosProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "!!Time %f: TestNotAlwaysAllZerosProbe::outputState failed for layer \"%s\"\n", timed, getTargetLayer()->getName());
      exit(EXIT_FAILURE);
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      if (nnz[b] != 0) {
         nonzeroValueOccurred = true;
      }
   }
   return status;
}

int TestNotAlwaysAllZerosProbe::initTestNotAlwaysAllZerosProbe_base() {
   nonzeroValueOccurred = false;
   return PV_SUCCESS;
}

int TestNotAlwaysAllZerosProbe::initTestNotAlwaysAllZerosProbe(const char * probeName, HyPerCol * hc) {
   return StatsProbe::initStatsProbe(probeName, hc);
}

void TestNotAlwaysAllZerosProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

BaseObject * createTestNotAlwaysAllZerosProbe(char const * name, HyPerCol * hc) {
   return hc ? new TestNotAlwaysAllZerosProbe(name, hc) : NULL;
}

}; // namespace PV
