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

pvdata_t L2NormFunction::evaluateLocal(float time, HyPerLayer * l) {
    pvdata_t l2norm = 0;
    pvdata_t * activityBuffer = l->getCLayer()->activity->data;
    int numNeurons = l->getNumNeurons();
    const int nx = l->getLayerLoc()->nx;
    const int ny = l->getLayerLoc()->ny;
    const int nf = l->getLayerLoc()->nf;
    const int nb = l->getLayerLoc()->nb;
    for(int n=0; n<numNeurons; n++) {
        int nex = kIndexExtended(n, nx, ny, nf, nb);
    	pvdata_t v = activityBuffer[nex];
        l2norm += v*v;
    }
    return 0.5*l2norm;
}  // end L2NormFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
