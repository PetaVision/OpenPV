/*
 * LogLatWTAFunction.cpp
 *
 *  Created on: Apr 26, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAFunction.hpp"

namespace PV {

LogLatWTAFunction::LogLatWTAFunction(const char * name) : LayerFunction(name) {
}  // end LogLatWTAFunction::LogLatWTAFunction(const char * name, HyPerLayer *)

pvdata_t LogLatWTAFunction::evaluateLocal(float time, HyPerLayer * l) {
    pvdata_t sum = 0;
    pvdata_t * activityBuffer = l->getCLayer()->activity->data;
    int numNeurons = l->getNumNeurons();
    const int nx = l->getLayerLoc()->nx;
    const int ny = l->getLayerLoc()->ny;
    const int nf = l->getLayerLoc()->nf;
    const int nb = l->getLayerLoc()->nb;
    for(int n=0; n<numNeurons; n+=nf /* assumes feature stride=1, x,y strides>1 */) {
        int nex = kIndexExtended(n, nx, ny, nf, nb);
    	pvdata_t * v = activityBuffer+nex;
    	pvdata_t aLa = 0;
    	for( int p=0; p<nf; p++) {
    	    for( int q=0; q<nf; q++ ) {
    	        aLa += p==q ? 0 : v[p]*v[q];
    	    }
    	}
        sum += log(1+aLa);
    }
    return sum;
}  // end LogLatWTAFunction::evaluate(float, HyPerLayer *)

}  // end namespace PV

