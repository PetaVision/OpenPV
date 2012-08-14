/*
 * ParameterSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "ParameterSweepTestProbe.hpp"

namespace PV {

ParameterSweepTestProbe::ParameterSweepTestProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   initParameterSweepTestProbe(filename, layer, msg);
}

ParameterSweepTestProbe::~ParameterSweepTestProbe() {
   // TODO Auto-generated destructor stub
}

int ParameterSweepTestProbe::initParameterSweepTestProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   int status = initStatsProbe(filename, layer, BufActivity, msg);
   HyPerLayer * l = getTargetLayer();
   const char * lname = l->getName();
   PVParams * params = l->getParent()->parameters();
   expectedSum = (double) params->value(lname, "expectedSum", 0.0f);
   expectedMin = params->value(lname, "expectedMin", 0.0f);
   expectedMax = params->value(lname, "expectedMax", 0.0f);
   return status;
}

int ParameterSweepTestProbe::outputState(float timef) {
   int status = StatsProbe::outputState(timef);
   if (timef >= 3.0 ) {
      assert(fabs(expectedSum - sum)<1e-6);
      assert(fabs(expectedMin - fMin)<1e-6);
      assert(fabs(expectedMax - fMax)<1e-6);
   }
   return status;
}

} /* namespace PV */
