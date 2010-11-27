/*
 * L2NormFunction.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "L2NormFunction.hpp"

namespace PV {

L2NormFunction::L2NormFunction(const char * name) : LayerFunction(name) {
}  // end L2NormFunction::L2NormFunction

pvdata_t L2NormFunction::evaluate(float time, HyPerLayer * l) {
    pvdata_t l2norm = 0;
    pvdata_t * Vbuffer = l->getV();
    pvdata_t numNeurons = l->getNumNeurons();
    for(int n=0; n<numNeurons; n++) {
    	pvdata_t v = Vbuffer[n];
        l2norm += v*v;
    }
    return 0.5*l2norm;
}  // end L2NormFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
