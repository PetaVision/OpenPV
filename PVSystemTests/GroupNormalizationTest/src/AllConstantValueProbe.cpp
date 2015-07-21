/*
 * AllConstantValueProbe.cpp
 */

#include <columns/HyPerCol.hpp>
#include "AllConstantValueProbe.hpp"

namespace PV {

AllConstantValueProbe::AllConstantValueProbe(char const * probeName, HyPerCol * hc) {
   initialize_base();
   initAllConstantValueProbe(probeName, hc);
}

AllConstantValueProbe::AllConstantValueProbe() {
   initialize_base();
}

int AllConstantValueProbe::initialize_base() {
   correctValue = (pvadata_t) 0;
   return PV_SUCCESS;
}

int AllConstantValueProbe::initAllConstantValueProbe(char const * probeName, HyPerCol * hc) {
   return StatsProbe::initStatsProbe(probeName, hc);
}

int AllConstantValueProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_correctValue(ioFlag);
   return status;
}

void AllConstantValueProbe::ioParam_correctValue(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "correctValue", &correctValue, correctValue/*default*/);
}

int AllConstantValueProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (this->parent->columnId()==0) {
      if (timed>0 && (fMin<correctValue-nnzThreshold || fMax > correctValue+nnzThreshold)) {
         fprintf(this->outputstream->fp, "     Values outside of tolerance nnzThreshold=%f\n", nnzThreshold);
         fprintf(stderr, "t=%f: fMin=%f, fMax=%f; values more than nnzThreshold=%g away from correct value %f\n", timed, fMin, fMax, nnzThreshold, correctValue);
         exit(EXIT_FAILURE);
      }
   }
   return status;
}

AllConstantValueProbe::~AllConstantValueProbe() {
}

} // namespace PV
