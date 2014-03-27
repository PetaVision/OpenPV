/*
 * RequireAllZeroActivityProbe.cpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 */

#include "RequireAllZeroActivityProbe.hpp"

namespace PV {

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initRequireAllZeroActivityProbe(probeName, hc);
}

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe() {
   initialize_base();
}

int RequireAllZeroActivityProbe::initialize_base() {
   nonzeroFound = false;
   nonzeroTime = 0.0;
   return PV_SUCCESS;
}

int RequireAllZeroActivityProbe::initRequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc) {
   int status = StatsProbe::initStatsProbe(probeName, hc);
   return status;
}

void RequireAllZeroActivityProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int RequireAllZeroActivityProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (nnz!=0) {
      if (!nonzeroFound) {
         nonzeroTime = timed;
      }
      nonzeroFound = true;
   }
   return status;
}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {
}

} /* namespace PV */
