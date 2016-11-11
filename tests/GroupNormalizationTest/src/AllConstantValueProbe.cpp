/*
 * AllConstantValueProbe.cpp
 */

#include "AllConstantValueProbe.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

AllConstantValueProbe::AllConstantValueProbe(char const *probeName, HyPerCol *hc) {
   initialize_base();
   initAllConstantValueProbe(probeName, hc);
}

AllConstantValueProbe::AllConstantValueProbe() { initialize_base(); }

int AllConstantValueProbe::initialize_base() {
   correctValue = (float)0;
   return PV_SUCCESS;
}

int AllConstantValueProbe::initAllConstantValueProbe(char const *probeName, HyPerCol *hc) {
   return StatsProbe::initStatsProbe(probeName, hc);
}

int AllConstantValueProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_correctValue(ioFlag);
   return status;
}

void AllConstantValueProbe::ioParam_correctValue(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValue(
         ioFlag, getName(), "correctValue", &correctValue, correctValue /*default*/);
}

int AllConstantValueProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (this->parent->columnId() == 0) {
      for (int b = 0; b < this->parent->getNBatch(); b++) {
         if (timed > 0
             && (fMin[b] < correctValue - nnzThreshold || fMax[b] > correctValue + nnzThreshold)) {
            outputStream->printf(
                  "     Values outside of tolerance nnzThreshold=%f\n", (double)nnzThreshold);
            ErrorLog().printf(
                  "t=%f: fMin=%f, fMax=%f; values more than nnzThreshold=%g away from correct "
                  "value %f\n",
                  timed,
                  (double)fMin[b],
                  (double)fMax[b],
                  (double)nnzThreshold,
                  (double)correctValue);
         }
      }
   }
   return status;
}

AllConstantValueProbe::~AllConstantValueProbe() {}

} // namespace PV
