/*
 * L2NormProbe.cpp
 *
 *  Created on: Nov 19, 2010
 *      Author: pschultz
 */

#include "L2NormProbe.hpp"

namespace PV {

L2NormProbe::L2NormProbe() : LayerFunctionProbe() {
   initL2NormProbe_base();
}

L2NormProbe::L2NormProbe(const char * probeName, HyPerCol * hc)
   : LayerFunctionProbe()
{
   initL2NormProbe_base();
   initL2NormProbe(probeName, hc);
}

L2NormProbe::~L2NormProbe() {
}

int L2NormProbe::initL2NormProbe(const char * probeName, HyPerCol * hc) {
   return initLayerFunctionProbe(probeName, hc);
}

void L2NormProbe::initFunction() {
   setFunction(new L2NormFunction(getName()));
}

int L2NormProbe::writeState(double timed, HyPerLayer * l, int batchIdx, pvdata_t value) {
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
   int nk = l->getNumGlobalNeurons();
   fprintf(outputstream->fp, "%st = %6.3f b = %d numNeurons = %8d L2-norm          = %f\n", getMessage(), timed, batchIdx, nk, value);
   fflush(outputstream->fp);
   return PV_SUCCESS;
}

}  // end namespace PV
