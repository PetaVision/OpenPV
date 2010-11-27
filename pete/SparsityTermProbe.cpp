/*
 * SparsityTermProbe.cpp
 *
 *  Created on: Nov 18, 2010
 *      Author: pschultz
 */

#include "SparsityTermProbe.hpp"

namespace PV {

SparsityTermProbe::SparsityTermProbe(const char * msg) : LayerFunctionProbe(msg) {
    function = new SparsityTermFunction(msg);
}
SparsityTermProbe::SparsityTermProbe(const char * filename, const char * msg) : LayerFunctionProbe(filename, msg) {
    function = new SparsityTermFunction(msg);
}

SparsityTermProbe::~SparsityTermProbe() {
    delete function;
}

int SparsityTermProbe::outputState(float time, HyPerLayer * l) {
    int nk = l->getNumNeurons();
    pvdata_t sum = function->evaluate(time, l);

    fprintf(fp, "%st = %6.3f numNeurons = %8d Sparsity Penalty = %f\n", msg, time, nk, sum);
    fflush(fp);

    return EXIT_SUCCESS;
}

}  // end of namespace PV
