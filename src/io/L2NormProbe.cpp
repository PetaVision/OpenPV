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

int L2NormProbe::outputState(float time, HyPerLayer * l) {
    int nk = l->getNumNeurons();
    pvdata_t l2norm = function->evaluate(time, l);

    fprintf(fp, "%st = %6.3f numNeurons = %8d L2-norm          = %f\n", msg, time, nk, l2norm);
    fflush(fp);

    return EXIT_SUCCESS;
}

}  // end namespace PV
