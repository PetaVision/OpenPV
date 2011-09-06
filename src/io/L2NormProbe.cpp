/*
 * L2NormProbe.cpp
 *
 *  Created on: Nov 19, 2010
 *      Author: pschultz
 */

#include "L2NormProbe.hpp"

namespace PV {

L2NormProbe::L2NormProbe(const char * msg) : LayerFunctionProbe(msg) {
   function = new L2NormFunction(msg);
}
L2NormProbe::L2NormProbe(const char * filename, HyPerCol * hc, const char * msg) : LayerFunctionProbe(filename, hc, msg) {
   function = new L2NormFunction(msg);
}

L2NormProbe::~L2NormProbe() {
   delete function;
}

int L2NormProbe::writeState(float time, HyPerLayer * l, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int nk = l->getNumGlobalNeurons();
   fprintf(fp, "%st = %6.3f numNeurons = %8d L2-norm          = %f\n", msg, time, nk, value);

   return PV_SUCCESS;
}

}  // end namespace PV
