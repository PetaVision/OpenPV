/*
 * LogLatWTAGenLayer.cpp
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAGenLayer.hpp"

namespace PV {

LogLatWTAGenLayer::LogLatWTAGenLayer(const char * name, HyPerCol * hc) :
      GenerativeLayer(name, hc)
{
   initialize_base();
   initialize();
}  // end of LogLatWTAGenLayer::LogLatWTAGenLayer(const char *, HyPerCol *)

LogLatWTAGenLayer::~LogLatWTAGenLayer() {
    free(dV);
}

int LogLatWTAGenLayer::initialize_base() {
    dV = NULL;
    return PV_SUCCESS;
}

int LogLatWTAGenLayer::initialize() {
    int nf = getLayerLoc()->nf;
    free(dV);
    dV = (pvdata_t *) malloc(nf*sizeof(pvdata_t));
    return dV != NULL ? PV_SUCCESS : PV_FAILURE;
}

int LogLatWTAGenLayer::updateV() {
    pvdata_t * V = getV();
    pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
    pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
    pvdata_t * GSynAux = this->getChannel(CHANNEL_INHB);
    // int nx = getLayerLoc()->nx;
    // int ny = getLayerLoc()->ny;
    int nf = getLayerLoc()->nf;
    for( int k=0; k<getNumNeurons(); k+=nf ) {
        // Assumes that stride in features is one.
    	pvdata_t sumacrossfeatures = 0;
    	for( int f=0; f<nf; f++) {
    	    sumacrossfeatures += V[k+f];
    	}
    	pvdata_t * Vthispos = V+k;
    	pvdata_t * GExcthispos = GSynExc+k;
        pvdata_t * GInhthispos = GSynInh+k;
        pvdata_t * GAuxthispos = GSynAux+k;
        pvdata_t latWTAexpr = latWTAterm(Vthispos,nf); // a'*Lslash*a
        for( int f=0; f<nf; f++) {
        	dV[f] = 2*(sumacrossfeatures-Vthispos[f])/(1+latWTAexpr);
        	Vthispos[f] += getRelaxation()*(GExcthispos[f]-GInhthispos[f]+auxChannelCoeff*GAuxthispos[f]-dV[f]);
            if(Vthispos[f] < 0) Vthispos[f] = 0;
        }
    }
    applyVMax();
    applyVThresh();
    return PV_SUCCESS;
}  // end of LogLatWTAGenLayer::updateV()

pvdata_t LogLatWTAGenLayer::latWTAterm(pvdata_t * V, int nf) {
	pvdata_t z=0;
    for( int p=0; p<nf; p++) {
    	for( int q=0; q<nf; q++) {
    	    z += p==q ? 0 : V[p]*V[q];
        }
    }
    return z;
}  // end of LogLatWTAGenLayer::latinhibsparsityterm(pvdata_t *, int)

}  // end of namespace PV block
