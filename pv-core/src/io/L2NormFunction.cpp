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

pvdata_t L2NormFunction::evaluateLocal(float time, HyPerLayer * l, int batchIdx) {
    double l2norm = 0;
    pvdata_t * activityBuffer = l->getCLayer()->activity->data + batchIdx * l->getNumExtended();
    int numNeurons = l->getNumNeurons();
    const int nx = l->getLayerLoc()->nx;
    const int ny = l->getLayerLoc()->ny;
    const int nf = l->getLayerLoc()->nf;
    const PVHalo * halo = &l->getLayerLoc()->halo;
    for(int n=0; n<numNeurons; n++) {
       int nex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
    	double v = (double) activityBuffer[nex];
        l2norm += v*v;
    }
    return (pvdata_t) (0.5*l2norm);
}  // end L2NormFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
