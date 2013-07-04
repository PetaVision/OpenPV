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

LogLatWTAProbe::LogLatWTAProbe() : LayerFunctionProbe() {
   initLogLatWTAProbe_base();
}

LogLatWTAProbe::LogLatWTAProbe(HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initLogLatWTAProbe_base();
   initLogLatWTAProbe(NULL, layer, msg);
}
LogLatWTAProbe::LogLatWTAProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initLogLatWTAProbe_base();
   initLogLatWTAProbe(filename, layer, msg);
}

LogLatWTAProbe::~LogLatWTAProbe() {
}

int LogLatWTAProbe::initLogLatWTAProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   LogLatWTAFunction * loglat = new LogLatWTAFunction(msg);
   return initLayerFunctionProbe(filename, layer, msg, loglat);
}

int LogLatWTAProbe::writeState(double timed, HyPerLayer * l, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int nk = l->getNumGlobalNeurons();
   fprintf(outputstream->fp, "%st = %6.3f numNeurons = %8d Lateral Competition Penalty = %f\n", msg, timed, nk, value);

   return PV_SUCCESS;
}

}  // end of namespace PV

