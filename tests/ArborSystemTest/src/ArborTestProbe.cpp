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

ArborTestProbe::ArborTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

ArborTestProbe::~ArborTestProbe() {}

int ArborTestProbe::initialize_base() { return PV_SUCCESS; }

int ArborTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void ArborTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      type             = BufActivity;
      PVParams *params = parent->parameters();
      if (params->present(name, "buffer")) {
         params->handleUnnecessaryStringParameter(name, "buffer");
         char const *buffer = params->stringValue(name, "buffer");
         FatalIf(!(buffer != NULL), "Test failed.\n");
         char *bufferlc = strdup(buffer);
         for (int c = 0; c < (int)strlen(bufferlc); c++) {
            bufferlc[c] = tolower(bufferlc[c]);
         }
         if (strcmp(bufferlc, "a") != 0 && strcmp(bufferlc, "activity") != 0) {
            if (parent->columnId() == 0) {
               ErrorLog().printf(
                     "   Value \"%s\" is inconsistent with correct value \"a\" or \"activity\".  "
                     "Exiting.\n",
                     buffer);
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
         free(bufferlc);
      }
   }
}

Response::Status ArborTestProbe::outputState(double timed) {
   auto status = StatsProbe::outputState(timed);
   if (status != Response::SUCCESS) {
      return status;
   }
   int const rank    = parent->getCommunicator()->commRank();
   int const rcvProc = 0;
   if (rank != rcvProc) {
      return status;
   }
   for (int b = 0; b < parent->getNBatch(); b++) {
      if (timed == 1.0) {
         FatalIf(!((avg[b] > 0.2499f) && (avg[b] < 0.2501f)), "Test failed.\n");
      }
      else if (timed == 2.0) {
         FatalIf(!((avg[b] > 0.4999f) && (avg[b] < 0.5001f)), "Test failed.\n");
      }
      else if (timed == 3.0) {
         FatalIf(!((avg[b] > 0.7499f) && (avg[b] < 0.7501f)), "Test failed.\n");
      }
      else if (timed > 3.0) {
         FatalIf(!((fMin[b] > 0.9999f) && (fMin[b] < 1.001f)), "Test failed.\n");
         FatalIf(!((fMax[b] > 0.9999f) && (fMax[b] < 1.001f)), "Test failed.\n");
         FatalIf(!((avg[b] > 0.9999f) && (avg[b] < 1.001f)), "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
