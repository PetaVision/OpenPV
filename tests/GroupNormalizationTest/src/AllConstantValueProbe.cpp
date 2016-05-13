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
      for(int b = 0; b < this->parent->getNBatch(); b++){
         if (timed>0 && (fMin[b]<correctValue-nnzThreshold || fMax[b] > correctValue+nnzThreshold)) {
            fprintf(this->outputstream->fp, "     Values outside of tolerance nnzThreshold=%f\n", nnzThreshold);
            fprintf(stderr, "t=%f: fMin=%f, fMax=%f; values more than nnzThreshold=%g away from correct value %f\n", timed, fMin[b], fMax[b], nnzThreshold, correctValue);
            exit(EXIT_FAILURE);
         }
      }
   }
   return status;
}

AllConstantValueProbe::~AllConstantValueProbe() {
}

BaseObject * createAllConstantValueProbe(char const * probeName, HyPerCol * hc) {
   return hc ? new AllConstantValueProbe(probeName, hc) : NULL;
}

} // namespace PV
