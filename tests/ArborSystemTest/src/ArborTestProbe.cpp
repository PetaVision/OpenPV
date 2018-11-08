/*
 * ArborTestProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "ArborTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

ArborTestProbe::ArborTestProbe(const char *name, PVParams *params, Communicator *comm)
      : StatsProbe() {
   initialize_base();
   initialize(name, params, comm);
}

ArborTestProbe::~ArborTestProbe() {}

int ArborTestProbe::initialize_base() { return PV_SUCCESS; }

void ArborTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void ArborTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      type             = BufActivity;
      PVParams *params = parameters();
      if (params->present(name, "buffer")) {
         params->handleUnnecessaryStringParameter(name, "buffer");
         char const *buffer = params->stringValue(name, "buffer");
         FatalIf(!(buffer != NULL), "Test failed.\n");
         char *bufferlc = strdup(buffer);
         for (int c = 0; c < (int)strlen(bufferlc); c++) {
            bufferlc[c] = tolower(bufferlc[c]);
         }
         if (strcmp(bufferlc, "a") != 0 && strcmp(bufferlc, "activity") != 0) {
            if (mCommunicator->commRank() == 0) {
               ErrorLog().printf(
                     "   Value \"%s\" is inconsistent with correct value \"a\" or \"activity\".  "
                     "Exiting.\n",
                     buffer);
            }
            MPI_Barrier(mCommunicator->communicator());
            exit(EXIT_FAILURE);
         }
         free(bufferlc);
      }
   }
}

Response::Status ArborTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   int const rank    = mCommunicator->commRank();
   int const rcvProc = 0;
   if (rank != rcvProc) {
      return status;
   }
   for (int b = 0; b < mLocalBatchWidth; b++) {
      if (simTime == 1.0) {
         FatalIf(!((avg[b] > 0.2499f) && (avg[b] < 0.2501f)), "Test failed.\n");
      }
      else if (simTime == 2.0) {
         FatalIf(!((avg[b] > 0.4999f) && (avg[b] < 0.5001f)), "Test failed.\n");
      }
      else if (simTime == 3.0) {
         FatalIf(!((avg[b] > 0.7499f) && (avg[b] < 0.7501f)), "Test failed.\n");
      }
      else if (simTime > 3.0) {
         FatalIf(!((fMin[b] > 0.9999f) && (fMin[b] < 1.001f)), "Test failed.\n");
         FatalIf(!((fMax[b] > 0.9999f) && (fMax[b] < 1.001f)), "Test failed.\n");
         FatalIf(!((avg[b] > 0.9999f) && (avg[b] < 1.001f)), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
