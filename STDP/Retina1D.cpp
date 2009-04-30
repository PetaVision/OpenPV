/*
 * Retina1DPattern.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: gkenyon
 */

#include "Retina1D.hpp"
#include <stdlib.h>
#include <assert.h>
#include "src/io/io.h"

namespace PV {

//Retina1D::Retina1D() : Retina() {
//}


Retina1D::Retina1D(const char * name, HyPerCol * hc) :
	Retina(name, hc) {
	targ1D = (pvdata_t *) malloc(sizeof(pvdata_t)*(clayer->numNeurons));
	createImage(clayer->V);
}

Retina1D::~Retina1D(){
	free(targ1D);
}


int Retina1D::createImage(pvdata_t * localBuf) {

	// local HyPerCube dimensions
	const int nx = clayer->loc.nx;
//	const int ny = clayer->loc.ny;
	const int nf = clayer->numFeatures;
	const int numNeurons = clayer->numNeurons;

	// global layer dimensions: every node processes whole image
	const int nxGlobal = clayer->loc.nxGlobal;
	const int nyGlobal = clayer->loc.nyGlobal;
	const int nfGlobal = nf; //clayer->numFeatures;
	const int numNeuronsGlobal = nxGlobal * nyGlobal * nfGlobal;
//	const int kx0 = clayer->loc.kx0; // assert(kx0 == 0);
//	const int ky0 = clayer->loc.ky0;
//	const int xMidGlobal = nxGlobal / 2.0;
	const int yMidGlobal = nyGlobal / 2.0;
	pvdata_t * globalBuf = (pvdata_t *) malloc(sizeof(pvdata_t) * numNeuronsGlobal);

	// only one MPI node creates global image
	if (clayer->columnId == 0) {

		// target is built from a specified number of
		// spaced segments separated by gaps
		// keeping the average density == clutterProb
		int numTargs = 2;  // TODO: read from params file
		int lenTarg = 8;   // TODO: read from params file
		float clutterProb = 0.5;     // TODO: read from params file
		lenTarg = (lenTarg > 0) ? lenTarg : 1; //can't have lenTarg == 0
		float clutterProbMin = ((numTargs * lenTarg) > 0) ? (numTargs * lenTarg
				/ (float) nxGlobal) : clutterProb;
		clutterProb = (clutterProb < clutterProbMin) ? clutterProbMin : clutterProb;

		// default all pixels to clutterProb
		for (long kg = 0; kg < numNeuronsGlobal; kg++) {
			globalBuf[kg] = rand() < (int) ( clutterProb * RAND_MAX );
		}

		// set lenGap so that average density equals clutterProb
		int lenGap = (clutterProb > 0) ? (lenTarg * (1.0 - clutterProb)
				/ clutterProb) : 0;

		// start 1D target at random location (in global coordinates)
		int beginTargX = rand() % ( nxGlobal - numTargs * (lenGap + lenTarg) );
		int beginTargY = yMidGlobal;
		int beginTargF = 1;

		// make each target segment + gap
		for (int iSeg = 0; iSeg < numTargs; iSeg++) {
			int startSegX = iSeg * (lenTarg + lenGap) + beginTargX;
			int endSegX = startSegX + lenTarg - 1;
			for (int iTargX = startSegX; iTargX < endSegX; iTargX++) {
				int iTargK = kIndex(iTargX, beginTargY, beginTargF, nxGlobal,
						nyGlobal, nfGlobal);
				globalBuf[iTargK] = 1;
			}
			int startGapX = endSegX + 1;
			int endGapX = startGapX + lenGap - 1;
			for (int iGapX = startGapX; iGapX < endGapX; iGapX++) {
				int iGapK = kIndex(iGapX, beginTargY, beginTargF, nxGlobal,
						nyGlobal, nfGlobal);
				globalBuf[iGapK] = 0;
			}
		}

		// if nf == 2 -> f[0] are OFF, f[1] are ON cells
		assert(nf == 1 || nf == 2);
		if (nf == 2) {
			for (int kg = 0; kg < nxGlobal * nyGlobal; kg++) {
				globalBuf[2 * kg] = globalBuf[kg];
				globalBuf[2 * kg + 1] = 0;
			}
		}
	}  // end if (l->columnId == 0)

	scatterReadBuf(this->clayer, globalBuf, localBuf, 0);
	for (int k = 0; k < numNeurons; k++) {
		targ1D[k] = localBuf[k];
	}
	free(globalBuf);

	return 0;
}

}
