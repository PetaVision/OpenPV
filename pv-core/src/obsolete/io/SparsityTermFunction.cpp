/*
 * SparsityTermFunction.cpp
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
    pvdata_t * activityBuffer = l->getCLayer()->activity->data;
    int numNeurons = l->getNumNeurons();
    const int nx = l->getLayerLoc()->nx;
    const int ny = l->getLayerLoc()->ny;
    const int nf = l->getLayerLoc()->nf;
    const PVHalo * halo = &l->getLayerLoc()->halo;
    for(int n=0; n<numNeurons; n++) {
        int nex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
    	pvdata_t v = activityBuffer[nex];
        sum += log(1+v*v);
    }
    return sum;
}  // end SparsityTermFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV
