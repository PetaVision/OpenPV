/*
 * TestAllZerosProbe.cpp
 *
 * A subclass of StatsProbe that verifies a layer is all zeros.
 *
 *  Created on: Dec 6, 2011
 *      Author: pschultz
 */

#include "TestAllZerosProbe.hpp"

namespace PV {

TestAllZerosProbe::TestAllZerosProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   initTestAllZerosProbe_base();
   initTestAllZerosProbe(filename, layer, msg);
}

TestAllZerosProbe::TestAllZerosProbe(HyPerLayer * layer, const char * msg) {
   initTestAllZerosProbe_base();
   initTestAllZerosProbe(NULL, layer, msg);
}

int TestAllZerosProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "!!Time %f: TestAllZerosProbe::outputState failed for layer \"%s\"\n", timed, getTargetLayer()->getName());
      exit(EXIT_FAILURE);
   }
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   if (nnz != 0) {
      fprintf(stderr, "!!Time %f: layer \"%s\" is not all zeroes.\n", timed, getTargetLayer()->getName());
      exit(EXIT_FAILURE);
   }
   return status;
}

int TestAllZerosProbe::initTestAllZerosProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   return StatsProbe::initStatsProbe(filename, layer, BufActivity, msg);
}

}; // namespace PV