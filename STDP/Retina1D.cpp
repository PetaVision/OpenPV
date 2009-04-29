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
	targ1D = (pvdata_t *) malloc(sizeof(clayer->numNeurons));
	createImage(clayer->V);
}

Retina1D::~Retina1D(){
	free(targ1D);
}


int Retina1D::createImage(pvdata_t * localBuf) {

	// local HyPerCube dimensions
	const int nx = clayer->loc.nx;
	const int ny = clayer->loc.ny;
	const int nf = clayer->numFeatures;

	// only one MPI node creates global image
	if (l->columnId == 0) {

		// global layer dimensions
		float nxGlobal = clayer->loc.nxGlobal;
		float nyGlobal = clayer->loc.nyGlobal;
		float kx0 = clayer->loc.kx0;
		float ky0 = clayer->loc.ky0;
		float xMidGlobal = nxGlobal / 2.0;
		float yMidGlobal = nyGlobal / 2.0;

		// target is built from a specified number of
		// spaced segments separated by gaps
		// keeping the average density == clutterProb
		int numTargs = 2;  // TODO: read from params file
		int lenTarg = 8;   // TODO: read from params file
		lenTarg = (lenTarg > 0) ? lenTarg : 1; //can't have lenTarg == 0
		float clutterProb = 0.5;     // TODO: read from params file
		float clutterProbMin = ((numTargs * lenTarg) > 0) ? (numTargs * lenTarg
				/ (float) nx) : clutterProb;
		clutterProb = (clutterProb < clutterProbMin) ? clutterProbMin : clutterProb;
		// default all pixels to clutterProb
		for (int k = 0; k < clayer->numNeurons; k++) {
			localBuf[k] = rand() < (int) ( clutterProb * RAND_MAX );
			targ1D[k] = 0;
		}



		// set lenGap so that average density equals clutterProb
		int lenGap = (clutterProb > 0) ? (lenTarg * (1.0 - clutterProb)
				/ clutterProb) : 0;
		int beginTarg = rand() % ( nx - numTargs * (lenGap + lenTarg) );

		for (int iSeg = 0; iSeg < numTargs; iSeg++) {
			int startSeg = iSeg * (lenTarg + lenGap) + beginTarg;
			int endSeg = startSeg + lenTarg - 1;
			for (int iTarg = startSeg; iTarg < endSeg; iTarg++) {
				localBuf[iTarg] = 1;
				targ1D[iTarg] = 1;
			}
			int startGap = endSeg + 1;
			int endGap = startGap + lenGap - 1;
			for (int iGap = startGap; iGap < endGap; iGap++) {
				localBuf[iGap] = 0;
			}
		}
		// f[0] are OFF, f[1] are ON cells
		assert(nf == 1 || nf == 2);
		if (nf == 2) {
			for (int k = 0; k < nx * ny; k++) {
				localBuf[2 * k] = localBuf[k];
				localBuf[2 * k + 1] = 0;
			}
		}
	}

	scatterReadBuf(l, globalBuf, localBuf, comm);



	return 0;
}

}
