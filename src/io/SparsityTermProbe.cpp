/*
 * SparsityTermProbe.cpp
 *
 *  Created on: Nov 18, 2010
 *      Author: pschultz
 */

#include "SparsityTermProbe.hpp"

namespace PV {

SparsityTermProbe::SparsityTermProbe(HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initSparsityTermProbe(NULL, layer, msg);
}

SparsityTermProbe::SparsityTermProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : LayerFunctionProbe()
{
   initSparsityTermProbe(filename, layer, msg);
}

SparsityTermProbe::~SparsityTermProbe() {
}

int SparsityTermProbe::initSparsityTermProbe(const char * filename, HyPerLayer * layer, const char * msg) {
   SparsityTermFunction * sparsity = new SparsityTermFunction(msg);
   return initLayerFunctionProbe(filename, layer, msg, sparsity);
}

int SparsityTermProbe::outputState(double timef) {
   HyPerLayer * l = getTargetLayer();
   int nk = l->getNumNeurons();
   pvdata_t sum = function->evaluate(timef, l);

   fprintf(fp, "%st = %6.3f numNeurons = %8d Sparsity Penalty = %f\n", msg, timef, nk, sum);
   fflush(fp);

   return EXIT_SUCCESS;
}

}  // end of namespace PV
