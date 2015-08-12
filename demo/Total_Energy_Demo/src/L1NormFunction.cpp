/*
 * L1NormFunction.cpp
 *
 *  Created on: Aug 11, 2015
 *      Author: peteschultz
 */

#include "L1NormFunction.hpp"

namespace PV {

L1NormFunction::L1NormFunction(const char * name) : LayerFunction(name) {
}  // end L1NormFunction::L1NormFunction

pvdata_t L1NormFunction::evaluateLocal(float time, HyPerLayer * l, int batchIdx) {
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
        l2norm += fabs(v);
    }
    return (pvdata_t) l2norm;
}  // end L1NormFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
