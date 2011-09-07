/*
 * SigmaPiLayer.cpp
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#include "CliqueLayer.hpp"
#include "..PetaVision/src/utils/conversions/h"
#inclue <assert.h>

namespace PV {

CliqueLayer::CliqueLayer() {
}

CliqueLayer::~CliqueLayer() {
}

CliqueLayer::CliqueLayer(const char* name, HyPerCol * hc) :
		ANNLayer(name, hc) {
}

CliqueLayer::CliqueLayer(const char* name, HyPerCol * hc, PVLayerType type) :
		ANNLayer(name, hc) {
}

int CliqueLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity,
		int neighbor) {
	recvsyn_timer->start();

	assert(neighbor >= 0);
	const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
	fflush(stdout);
#endif

	// get margin indices (should compute these in HyPerLayer::initialize
	//unsigned int kPreExt = 0;
	unsigned int kMargin = 0;
	unsigned int marginUp = this->clayer->loc.halo.up;
	unsigned int marginDn = this->clayer->loc.halo.dn;
	unsigned int marginLt = this->clayer->loc.halo.lt;
	unsigned int marginRt = this->clayer->loc.halo.rt;
	unsigned int numMargin = marginUp * marginDn * marginLt * marginRt;
	assert(numMargin == this->clayer->numExtended - this->clayer->numNeurons);
	unsigned int nf = this->clayer->loc.nf;
	unsigned int nx = this->clayer->loc.nx;
	unsigned int ny = this->clayer->loc.ny;
	unsigned int nxExt = nx + marginRt + marginLt;
	unsigned int nyExt = ny + marginUp + marginDn;
	unsigned int syExt = nf * nxExt;
	unsigned int sxExt = nf;
	unsigned int * marginIndices = (unsigned int *) calloc(numMargin,
			sizeof(int));
	assert(marginIndices != NULL);
	// get North margin indices
	for (unsigned int kPreExt = 0; kPreExt < nf * nxExt * marginUp;
			kPreExt++) {
		marginIndices[kMargin++] = kPreExt;
	}assert(kMargin == nf * nxExt * marginUp);
	// get East margin indices
	for (unsigned int ky = marginUp; ky < marginUp + ny; ky++) {
		for (unsigned int kx = 0; kx < marginLt; kx++) {
			for (unsigned int kf = 0; kf < nf; kf++) {
				unsigned int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
				marginIndices[kMargin++] = kPreExt;
			}
		}
	}assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny);
	// get West margin indices
	for (unsigned int ky = marginUp; ky < marginUp + ny; ky++) {
		for (unsigned int kx = nx + marginLt; kx < nxExt; kx++) {
			for (unsigned int kf = 0; kf < nf; kf++) {
				unsigned int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
				marginIndices[kMargin++] = kPreExt;
			}
		}
	}assert(
			kMargin == nf * nxExt * marginUp + nf * marginLt * ny + nf * marginUp * ny);
	// get South margin indices
	for (unsigned int kPreExt = kMargin; kPreExt < numMargin; kPreExt++) {
		marginIndices[kMargin++] = kPreExt;
	}assert(kMargin == numMargin);

	// gather active indices in extended layer
	// note: activeIndices currently only dimensioned to numNeurons and refers to global indices
	// TODO: make activeIndices local and convert to global in writeActivitySparse sequence, dimension activeIndices to numExtended
	// init activeIndicesExt to activeIndices and append indices of active neurons in margins to end of list
	int numActiveExt = clayer->numActive;
	unsigned int * activeExt = clayer->activeIndices;
	float * aPre = activity->data;
	for (kMargin = 0; kMargin < numMargin; kMargin++) {
		unsigned int kPreExt = marginIndices[kMargin];
		if (aPre[kPreExt] == 0)
			continue;
		activeExt[numActiveExt++] = kPreExt;
	}

	// calc dimensions of wPostPatch
	// TODO: following is copied from calculation of wPostPatches and should be pre-calculated and stored there in conn::wPostPatches
	// TODO: following should be implemented as HyPerConn::calcPostPatchSize
	const PVLayer * lPre = clayer;
	const int xScale = conn->post->getXScale() - this->getXScale();
	const int yScale = conn->post->getYScale() - this->getYScale();
	const float powXScale = powf(2.0f, (float) xScale);
	const float powYScale = powf(2.0f, (float) yScale);
	const int prePad = lPre->loc.nb;
	const int nxPostPatch = (int) (conn->nxp * powXScale); // TODO: store in HyPerConn::wPostPatches
	const int nyPostPatch = (int) (conn->nyp * powYScale);// TODO: store in HyPerConn::wPostPatches
	const int nfPostPatch = nf;

	// clique dimensions
	// a new set of cliques is centered on each pre-synaptic cell with radius nzPostPatch/2
	// TODO: precompute clique dimensions during initialize
	int nyCliqueRadius = (int) (nyPostPatch/2);
	int nxCliqueRadius = (int) (nxPostPatch/2);
	unsigned int cliquePatchSize = (2*nxCliqueRadius + 1) * (2*nyCliqueRadius + 1) * nf;
	unsigned int cliqueSize = 2;// number of presynaptic cells in clique (traditional ANN uses 1)
	unsigned int num_kernels = conn->numDataPatches(neighbor);
	unsigned int calc_kernels = conn->KernelConn::numDataPatches(neighbor) * pow(cliquePatchSize, cliqueSize-1);
	assert(num_kernels == calc_kernels);

	// loop over all products of cliqueSize active presynaptic cells
	// outer loop is over presynaptic cells, each of which defines the center of a cliquePatch
	// inner loop is over all combinations of clique cells within cliquePatch boundaries, which may be shrunken
	unsigned int * cliqueActiveIndices = calloc(cliquePatchSize, sizeof(unsigned int));
	for (int kPreActive = 0; kPreActive < numActiveExt; kPreActive++) {
		unsigned int kPreExt = activeExt[kPreActive];

		// get active presynaptic neuron indices in wPostPatch
		unsigned int numActiveElements = 0;
		int kxPreExt = kxPos(kPreExt, nxExt, nyExt, nf);
		int kyPreExt = kyPos(kPreExt, nxExt, nyExt, nf);
		int kfPreExt = featureIndex(kPreExt, nxExt, nyExt, nf);

		// get indices of active clique elements
		for(int kyCliqueExt = kyPreExt - nyCliqueRadius; kyCliqueExt < kyPreExt + nyCliqueRadius; kyCliqueExt++) {
			for(int kxCliqueExt = kxPreExt - nxCliqueRadius; kxCliqueExt < kxPreExt + nxCliqueRadius; kxCliqueExt++) {
				for(int kfCliqueExt = 0; kfCliqueExt < nf; kfCliqueExt++) {
					unsigned int kCliqueExt = kIndex(kxCliqueExt, kyCliqueExt, kfCliqueExt, nxExt, nyExt, nf);
					if aPre[kCliqueExt] == 0) continue;
					cliqueActiveIndices[numActiveElements++] = kCliqueExt;
				}
			}
		}

		// loop over all active elements in clique centered on kPreExt
		unsigned int numActiveCliques = pow(numActiveElements, cliqueSize-1);
		for(unsigned int kClique = 0; kClique < numActiveCliques; kClique++) {

			// decompose kClique to compute product of active clique elements
			unsigned int kernelNdx = 0;
			pvdata_t cliqueProd = aPre[kPreExt];
			unsigned int kResidue = kClique;
			for(unsigned int iProd = 0; iProd < cliqueSize-1; iProd++) {
				unsigned int kPatchActive = (unsigned int) (kResidue / numActiveElements);
				unsigned int kCliqueExt = cliqueActiveIndices[kPatchActive];
				cliqueProd *= aPre[kCliqueExt];
				kResidue = kResidue - kPatchActive * numActiveElements;

				// compute kernel_index for this clique element
				int kxCliqueExt = kxPos(kCliqueExt, nxExt, nyExt, nf);
				int kyCliqueExt = kyPos(kCliqueExt, nxExt, nyExt, nf);
				int kfClique = featureIndex(kCliqueExt, nxExt, nyExt, nf);
				int kxPatch = kxCliqueExt - kxPreExt + nxCliqueRadius;
				int kyPatch = kyCliqueExt - kyPreExt + nyCliqueRadius;
				unsigned int kKernel = kIndex(kxPatch, kyPatch, kfClique, (2*nxCliqueRadius + 1), (2*nyCliqueRadius + 1), nf);
				kernelNdx += kKernel * pow(cliquePatchSize,iProd);
			}

			// get weight for this clique
			PVPatch * w_patch = conn->getWeights(kPreExt,neighbor);


		} // kClique
	} // kPreActive
	delete(cliqueActiveIndices);

	for( int kPre=0; kPre < numExtended; kPre++) {
		float a = activity->data[kPre];

		// Activity < 0 is used by generative models --pete
		if (a == 0.0f) continue;// TODO - assume activity is sparse so make this common branch

		PVAxonalArbor * arbor = conn->axonalArbor(kPre, neighbor);
		PVPatch * GSyn = arbor->data;
		PVPatch * weights = arbor->weights;

		// WARNING - assumes weight and GSyn patches from task same size
		//         - assumes patch stride sf is 1

		int nk = GSyn->nf * GSyn->nx;
		int ny = GSyn->ny;
		int sy = GSyn->sy;// stride in layer
		int syw = weights->sy;// stride in patch

		// TODO - unroll
		for (int y = 0; y < ny; y++) {
			pvpatch_accumulate(nk, GSyn->data + y*sy, a, weights->data + y*syw);
//       if (err != 0) printf("  ERROR kPre = %d\n", kPre);
		}
	}

	recvsyn_timer->stop();

	return 0;
}

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

	return 0;
}

} /* namespace PV */

