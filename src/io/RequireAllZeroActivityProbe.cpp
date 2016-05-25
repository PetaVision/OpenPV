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
   exitOnFailure = false;
   return PV_SUCCESS;
}

int RequireAllZeroActivityProbe::initRequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc) {
   int status = StatsProbe::initStatsProbe(probeName, hc);
   return status;
}

int RequireAllZeroActivityProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
    int status = StatsProbe::ioParamsFillGroup(ioFlag);
    ioParam_exitOnFailure(ioFlag);
    return status;
}

void RequireAllZeroActivityProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

void RequireAllZeroActivityProbe::ioParam_exitOnFailure(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "exitOnFailure", &exitOnFailure, exitOnFailure);
}

int RequireAllZeroActivityProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   for(int b = 0; b < getParent()->getNBatch(); b++){
      if (nnz[b]!=0) {
         if (!nonzeroFound) {
            nonzeroTime = timed;
         }
         nonzeroFound = true;
      }
   }
   return status;
}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {
    //We check for exits on failure in destructor
    if(getNonzeroFound()){
        fprintf(stderr, "%s \"%s\" error: Nonzero activity found\n", getKeyword(), name);
        exit(EXIT_FAILURE);
    }
}

BaseObject * createRequireAllZeroActivityProbe(char const * name, HyPerCol * hc) {
   return hc ? new RequireAllZeroActivityProbe(name, hc) : NULL;
}

} /* namespace PV */
