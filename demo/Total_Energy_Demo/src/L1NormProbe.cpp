/*
 * L1NormProbe.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#include "L1NormProbe.hpp"

namespace PV {

L1NormProbe::L1NormProbe() : LayerFunctionProbe() {
   initL1NormProbe_base();
}

L1NormProbe::L1NormProbe(const char * probeName, HyPerCol * hc)
   : LayerFunctionProbe()
{
   initL1NormProbe_base();
   initL1NormProbe(probeName, hc);
}

L1NormProbe::~L1NormProbe() {
}

int L1NormProbe::initL1NormProbe(const char * probeName, HyPerCol * hc) {
   return initLayerFunctionProbe(probeName, hc);
}

void L1NormProbe::initFunction() {
   setFunction(new L1NormFunction(getName()));
}

int L1NormProbe::writeState(double timed, HyPerLayer * l, int batchIdx, pvdata_t value) {
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
   int nk = l->getNumGlobalNeurons();
   fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d L1-norm          = %f\n", getMessage(), timed, batchIdx, nk, value);
   fflush(outputstream->fp);
   return PV_SUCCESS;
}

}  // end namespace PV
