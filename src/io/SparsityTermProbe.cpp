/*
 * SparsityTermProbe.cpp
 *
 *  Created on: Nov 18, 2010
 *      Author: pschultz
 */

#include "SparsityTermProbe.hpp"

namespace PV {

SparsityTermProbe::SparsityTermProbe() {
   initSparsityTermProbe_base();
}

SparsityTermProbe::SparsityTermProbe(const char * probeName, HyPerCol * hc)
   : LayerFunctionProbe()
{
   initSparsityTermProbe_base();
   initSparsityTermProbe(probeName, hc);
}

SparsityTermProbe::~SparsityTermProbe() {
}

int SparsityTermProbe::initSparsityTermProbe(const char * probeName, HyPerCol * hc) {
   return initLayerFunctionProbe(probeName, hc);
}

void SparsityTermProbe::initFunction() {
   setFunction(new SparsityTermFunction(getName()));
}

int SparsityTermProbe::outputState(double timef) {
   HyPerLayer * l = getTargetLayer();
   int nk = l->getNumNeurons();
   pvdata_t sum = getFunction()->evaluate(timef, l);

   if (outputstream && outputstream->fp) {
      fprintf(outputstream->fp, "%st = %6.3f numNeurons = %8d Sparsity Penalty = %f\n", getMessage(), timef, nk, sum);
      fflush(outputstream->fp);
   }

   return EXIT_SUCCESS;
}

}  // end of namespace PV
