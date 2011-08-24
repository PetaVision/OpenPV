/*
 * LogLatWTAProbe.cpp
 *
 * A derived class of LayerFunctionProbe that uses LogLatWTAFunction
 *
 *  Created on: Apr 26, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAProbe.hpp"

namespace PV {

LogLatWTAProbe::LogLatWTAProbe(const char * msg) : LayerFunctionProbe(msg) {
   function = new LogLatWTAFunction(msg);
}
LogLatWTAProbe::LogLatWTAProbe(const char * filename, HyPerCol * hc, const char * msg) : LayerFunctionProbe(filename, hc, msg) {
   function = new LogLatWTAFunction(msg);
}

LogLatWTAProbe::~LogLatWTAProbe() {
   delete function;
}

int LogLatWTAProbe::writeState(float time, HyPerLayer * l, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int nk = l->getNumNeurons();
   fprintf(fp, "%st = %6.3f numNeurons = %8d Lateral Competition Penalty = %f\n", msg, time, nk, value);
   fflush(fp);

   return PV_SUCCESS;
}

}  // end of namespace PV

