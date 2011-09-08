/*
 * CliqueLayer.cpp
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#include "CliqueLayer.hpp"
#include "../utils/conversions.h"
#include <assert.h>

namespace PV {

CliqueLayer::CliqueLayer(const char* name, HyPerCol * hc) :
		ANNLayer(name, hc) {
}

CliqueLayer::CliqueLayer(const char* name, HyPerCol * hc, PVLayerType type) :
		ANNLayer(name, hc) {
}


int CliqueLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity,
		int neighbor) {
	recvsyn_timer->start();

	assert(neighbor == 0); // assume called only once
	//const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
	fflush(stdout);
#endif

	// get margin indices (should compute these in HyPerLayer::initialize
	//unsigned int kPreExt = 0;
	int kMargin = 0;
	int marginUp = this->clayer->loc.halo.up;
	int marginDn = this->clayer->loc.halo.dn;
	int marginLt = this->clayer->loc.halo.lt;
	int marginRt = this->clayer->loc.halo.rt;
	int numMargin = marginUp * marginDn * marginLt * marginRt;
	assert(numMargin == this->getNumExtended() - this->getNumNeurons());
	int nf = this->clayer->loc.nf;
	int nx = this->clayer->loc.nx;
	int ny = this->clayer->loc.ny;
	int nxExt = nx + marginRt + marginLt;
	int nyExt = ny + marginUp + marginDn;
	//int syExt = nf * nxExt;
	//int sxExt = nf;
	int * marginIndices = (int *) calloc(numMargin,
			sizeof(int));
	assert(marginIndices != NULL);
	// get North margin indices
	for (int kPreExt = 0; kPreExt < nf * nxExt * marginUp; kPreExt++) {
		marginIndices[kMargin++] = kPreExt;
	}
	assert(kMargin == nf * nxExt * marginUp);
	// get East margin indices
	for (int ky = marginUp; ky < marginUp + ny; ky++) {
		for (int kx = 0; kx < marginLt; kx++) {
			for (int kf = 0; kf < nf; kf++) {
				int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
				marginIndices[kMargin++] = kPreExt;
			}
		}
	}
	assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny);
	// get West margin indices
	for (int ky = marginUp; ky < marginUp + ny; ky++) {
		for (int kx = nx + marginLt; kx < nxExt; kx++) {
			for (int kf = 0; kf < nf; kf++) {
				int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
				marginIndices[kMargin++] = kPreExt;
			}
		}
	}
	assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny + nf * marginUp * ny);
	// get South margin indices
	for (int kPreExt = kMargin; kPreExt < numMargin; kPreExt++) {
		marginIndices[kMargin++] = kPreExt;
	}
	assert(kMargin == numMargin);

	// gather active indices in extended layer
	// init activeIndicesExt to activeIndices and append indices of active neurons in margins to end of list
	int numActiveExt = clayer->numActive;
	unsigned int * activeExt = clayer->activeIndices;
	float * aPre = activity->data;
	for (kMargin = 0; kMargin < numMargin; kMargin++) {
		int kPreExt = marginIndices[kMargin];
		if (aPre[kPreExt] == 0)
			continue;
		activeExt[numActiveExt++] = kPreExt;
	}

	// calc dimensions of wPostPatch
	// TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in HyPerConn::wPostPatches
	// TODO: following should be implemented as HyPerConn::calcPostPatchSize
	//const PVLayer * lPre = clayer;
	const float xScaleDiff = conn->getPost()->getXScale() - getXScale();
	const float yScaleDiff = conn->getPost()->getYScale() - getYScale();
	const float powXScale = powf(2.0f, (float) xScaleDiff);
	const float powYScale = powf(2.0f, (float) yScaleDiff);
	//const int prePad = lPre->loc.nb;
	const int nxPostPatch = (int) (conn->xPatchSize() * powXScale); // TODO: store in HyPerConn::wPostPatches
	const int nyPostPatch = (int) (conn->yPatchSize() * powYScale);// TODO: store in HyPerConn::wPostPatches
	//const int nfPostPatch = nf;

	// clique dimensions
	// a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
	// TODO: precompute clique dimensions during CliqueConn::initialize
	int nyCliqueRadius = (int) (nyPostPatch/2);
	int nxCliqueRadius = (int) (nxPostPatch/2);
	int cliquePatchSize = (2*nxCliqueRadius + 1) * (2*nyCliqueRadius + 1) * nf;
	int cliqueSize = 2;// number of presynaptic cells in clique (traditional ANN uses 1)
	//int numKernels = conn->numDataPatches();  // per arbor?
	int numCliques = pow(cliquePatchSize, cliqueSize-1);
	assert(numCliques == conn->numberOfAxonalArborLists());

	// loop over all products of cliqueSize active presynaptic cells
	// outer loop is over presynaptic cells, each of which defines the center of a cliquePatch
	// inner loop is over all combinations of clique cells within cliquePatch boundaries, which may be shrunken
	// TODO: pre-allocate cliqueActiveIndices as CliqueConn::cliquePatchSize member variable
	int * cliqueActiveIndices = (int *) calloc(cliquePatchSize, sizeof(int));
	assert(cliqueActiveIndices != NULL);
	for (int kPreActive = 0; kPreActive < numActiveExt; kPreActive++) {
		int kPreExt = activeExt[kPreActive];

                // get indices of active elements in clique radius
		int numActiveElements = 0;
		int kxPreExt = kxPos(kPreExt, nxExt, nyExt, nf);
		int kyPreExt = kyPos(kPreExt, nxExt, nyExt, nf);
		//int kfPreExt = featureIndex(kPreExt, nxExt, nyExt, nf);
		for(int kyCliqueExt = kyPreExt - nyCliqueRadius; kyCliqueExt < kyPreExt + nyCliqueRadius; kyCliqueExt++) {
			for(int kxCliqueExt = kxPreExt - nxCliqueRadius; kxCliqueExt < kxPreExt + nxCliqueRadius; kxCliqueExt++) {
				for(int kfCliqueExt = 0; kfCliqueExt < nf; kfCliqueExt++) {
					int kCliqueExt = kIndex(kxCliqueExt, kyCliqueExt, kfCliqueExt, nxExt, nyExt, nf);
					if (aPre[kCliqueExt] == 0) continue;
					cliqueActiveIndices[numActiveElements++] = kCliqueExt;
				}
			}
		}

		// loop over all active elements in clique radius
		int numActiveCliques = pow(numActiveElements, cliqueSize-1);
		for(int kClique = 0; kClique < numActiveCliques; kClique++) {

			// decompose kClique to compute product of active clique elements
			int arborNdx = 0;
			pvdata_t cliqueProd = aPre[kPreExt];
			int kResidue = kClique;
			for(int iProd = 0; iProd < cliqueSize-1; iProd++) {
				int kPatchActive = (unsigned int) (kResidue / numActiveElements);
				int kCliqueExt = cliqueActiveIndices[kPatchActive];
				cliqueProd *= aPre[kCliqueExt];
				kResidue = kResidue - kPatchActive * numActiveElements;

				// compute arborIndex for this clique element
				int kxCliqueExt = kxPos(kCliqueExt, nxExt, nyExt, nf);
				int kyCliqueExt = kyPos(kCliqueExt, nxExt, nyExt, nf);
				int kfClique = featureIndex(kCliqueExt, nxExt, nyExt, nf);
				int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
				int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
				unsigned int kArbor = kIndex(kxPatch, kyPatch, kfClique, (2*nxCliqueRadius + 1), (2*nyCliqueRadius + 1), nf);
				arborNdx += kArbor * pow(cliquePatchSize,iProd);
			}

			// receive weights input from clique (mostly copied from superclass method)
	                PVAxonalArbor * arbor = conn->axonalArbor(kPreExt, arborNdx);
	                PVPatch * GSyn = arbor->data;
	                PVPatch * weights = arbor->weights;

	                // WARNING - assumes weight and GSyn patches from task same size
	                //         - assumes patch stride sf is 1

	                int nkPost = GSyn->nf * GSyn->nx;
	                //int nyPost = GSyn->ny;
	                int syPost = GSyn->sy;// stride in layer
	                int sywPatch = weights->sy;// stride in patch

	                // TODO - unroll
	                for (int y = 0; y < ny; y++) {
	                        pvpatch_accumulate(nkPost, GSyn->data + y*syPost, cliqueProd, weights->data + y*sywPatch);
	                }

		} // kClique
	} // kPreActive
	delete(cliqueActiveIndices);
	recvsyn_timer->stop();
	return 0;
}

// TODO: direct clique input to separate GSyn: CHANNEL_CLIQUE
// the following is copied directly from ODDLayer::updateState()
int CliqueLayer::updateState(float time, float dt) {

	pv_debug_info("[%d]: CliqueLayer::updateState:", clayer->columnId);

	pvdata_t * V = clayer->V;
	pvdata_t * phiExc = getChannel(CHANNEL_EXC);
	pvdata_t * phiInh = getChannel(CHANNEL_INH);

	// assume bottomUp input to phiExc, lateral input to phiInh
	for (int k = 0; k < clayer->numNeurons; k++) {
		pvdata_t bottomUp_input = phiExc[k];
		pvdata_t lateral_input = phiInh[k];
		V[k] = (bottomUp_input > 0.0f) ?
				bottomUp_input * lateral_input : bottomUp_input;
	}

	resetGSynBuffers();
	applyVMax();
	applyVThresh();
	setActivity();
	updateActiveIndices();

	return 0;
}

int CliqueLayer::updateActiveIndices(){
   int numActive = 0;
   PVLayerLoc & loc = clayer->loc;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < getNumNeurons(); k++) {
      const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.nb);
      if (activity[kex] > 0.0) {
         clayer->activeIndices[numActive++] = k; //globalIndexFromLocal(k, loc);
      }
   }
   clayer->numActive = numActive;
   return PV_SUCCESS;
}

} /* namespace PV */


