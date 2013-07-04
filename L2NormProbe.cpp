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

L2NormProbe::L2NormProbe(HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initL2NormProbe_base();
   initL2NormProbe(NULL, layer, msg);
}

L2NormProbe::L2NormProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initL2NormProbe_base();
   initL2NormProbe(filename, layer, msg);
}

L2NormProbe::~L2NormProbe() {
}

int L2NormProbe::initL2NormProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   L2NormFunction * l2norm = new L2NormFunction(msg);
   return initLayerFunctionProbe(filename, layer, msg, l2norm);
}

int L2NormProbe::writeState(double timed, HyPerLayer * l, pvdata_t value) {
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
   int nk = l->getNumGlobalNeurons();
   fprintf(outputstream->fp, "%st = %6.3f numNeurons = %8d L2-norm          = %f\n", msg, timed, nk, value);
   fflush(outputstream->fp);
   return PV_SUCCESS;
}

}  // end namespace PV
