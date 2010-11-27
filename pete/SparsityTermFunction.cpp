/*
 * L2NormFunction.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "SparsityTermFunction.hpp"

namespace PV {

SparsityTermFunction::SparsityTermFunction(const char * name) : LayerFunction(name) {
}  // end SparsityTermFunction::SparsityTermFunction(const char * name, HyPerLayer *)

pvdata_t SparsityTermFunction::evaluate(float time, HyPerLayer * l) {
    pvdata_t sum = 0;
    pvdata_t * Vbuffer = l->getV();
    int numNeurons = l->getNumNeurons();
    for(int n=0; n<numNeurons; n++) {
    	pvdata_t v = Vbuffer[n];
        sum += log(1+v*v);
    }
    return sum;
}  // end SparsityTermFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
