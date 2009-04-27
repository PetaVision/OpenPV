/*
 * Retina1DPattern.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: gkenyon
 */

#include "Retina1D.hpp"
#include <stdlib.h>
#include <assert.h>

namespace PV {

//Retina1D::Retina1D() : Retina() {
//}


Retina1D::Retina1D(const char * name, HyPerCol * hc) :
	Retina(name, hc) {
	targ1D = (pvdata_t *) malloc(sizeof(clayer->numNeurons));;
	createImage(clayer->V);
}

Retina1D::~Retina1D(){
	free(targ1D);
}


int Retina1D::createImage(pvdata_t * buf) {
	const int nx = clayer->loc.nx;
	const int ny = clayer->loc.ny;
	const int nf = clayer->numFeatures;
	// target is built from a specified number of
	// spaced segments separated by gaps
	// keeping the average density == clutterProb
	int numTargs = 2;
	int lenTarg = 8;
	lenTarg = (lenTarg > 0) ? lenTarg : 1; //can't have lenTarg == 0
	float clutterProb = 0.5;
	float clutterProbMin = ((numTargs * lenTarg) > 0) ? (numTargs * lenTarg
			/ (float) nx) : clutterProb;
	clutterProb = (clutterProb < clutterProbMin) ? clutterProbMin : clutterProb;
	// set lenGap so that average density equals clutterProb
	int lenGap = (clutterProb > 0) ? (lenTarg * (1.0 - clutterProb)
			/ clutterProb) : 0;
	int beginTarg = rand() % ( nx - numTargs * (lenGap + lenTarg) )  ;
	for (int k = 0; k < clayer->numNeurons; k++) {
		buf[k] = rand() < (int) ( clutterProb * RAND_MAX );
		targ1D[k] = 0;
	}
	for (int iSeg = 0; iSeg < numTargs; iSeg++) {
		int startSeg = iSeg * (lenTarg + lenGap) + beginTarg;
		int endSeg = startSeg + lenTarg - 1;
		for (int iTarg = startSeg; iTarg < endSeg; iTarg++) {
			buf[iTarg] = 1;
			targ1D[iTarg] = 1;
		}
		int startGap = endSeg + 1;
		int endGap = startGap + lenGap - 1;
		for (int iGap = startGap; iGap < endGap; iGap++) {
			buf[iGap] = 0;
		}
	}
	// f[0] are OFF, f[1] are ON cells
	assert(nf == 1 || nf == 2);
	if (nf == 2) {
		for (int k = 0; k < nx * ny; k++) {
			buf[2 * k] = buf[k];
			buf[2 * k + 1] = 0;
		}
	}

	return 0;
}

}
