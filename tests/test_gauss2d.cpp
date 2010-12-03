/**
 * This file tests weight initialization to a 2D Gaussian with sigma = 1.0 and normalized to 1.0
 * Test compares HyPerConn to KernelConn,
 * assumes kernelConn produces correct 2D Gaussian weights
 *
 */

#undef DEBUG_PRINT

#include "../src/layers/HyPerLayer.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/layers/Example.hpp"
#include "../src/io/io.h"
#include <assert.h>

using namespace PV;

int check_kernel_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre,
		int axonID);

int main(int argc, char * argv[]) {
	PV::HyPerCol * hc = new PV::HyPerCol("test_gauss2d column", argc, argv);
	PV::Example * pre = new PV::Example("test_gauss2d pre", hc);
	PV::Example * post = new PV::Example("test_gauss2d post", hc);
	PV::HyPerConn * cHyPer = new HyPerConn("test_gauss2d hyperconn", hc, pre,
			post, CHANNEL_EXC);
	PV::KernelConn * cKernel = new KernelConn("test_gauss2d kernelconn", hc,
			pre, post, CHANNEL_EXC);
	PV::Example * pre2 = new PV::Example("test_gauss2d pre 2", hc);
	PV::Example * post2 = new PV::Example("test_gauss2d post 2", hc);
	PV::HyPerConn * cHyPer1to2 = new HyPerConn("test_gauss2d hyperconn 1 to 2", hc, pre,
			post2, CHANNEL_EXC);
	PV::KernelConn * cKernel1to2 = new KernelConn("test_gauss2d kernelconn 1 to 2", hc,
			pre, post2, CHANNEL_EXC);
	PV::HyPerConn * cHyPer2to1 = new HyPerConn("test_gauss2d hyperconn 2 to 1", hc, pre2,
			post, CHANNEL_EXC);
	PV::KernelConn * cKernel2to1 = new KernelConn("test_gauss2d kernelconn 2 to 1", hc,
			pre2, post, CHANNEL_EXC);

	const int axonID = 0;
	int num_pre_extended = pre->clayer->numExtended;
	assert(num_pre_extended == cHyPer->numWeightPatches(axonID));

	int status = 0;
	for (int kPre = 0; kPre < num_pre_extended; kPre++) {
		status = check_kernel_vs_hyper(cHyPer, cKernel, kPre, axonID);
		assert(status==0);
		status = check_kernel_vs_hyper(cHyPer1to2, cKernel1to2, kPre, axonID);
		assert(status==0);
		status = check_kernel_vs_hyper(cHyPer2to1, cKernel2to1, kPre, axonID);
		assert(status==0);
	}

	delete hc;
	return 0;
}

int check_kernel_vs_hyper(HyPerConn * cHyPer, KernelConn * cKernel, int kPre,
		int axonID) {
	int status = 0;
	PVPatch * hyperPatch = cHyPer->getWeights(kPre, axonID);
	PVPatch * kernelPatch = cKernel->getWeights(kPre, axonID);

	int nk = hyperPatch->nf * hyperPatch->nx;
	assert(nk == (kernelPatch->nf * kernelPatch->nx));
	int ny = hyperPatch->ny;
	assert(ny == kernelPatch->ny);
	int sy = hyperPatch->sy;
	assert(sy == kernelPatch->sy);
	pvdata_t * hyperWeights = hyperPatch->data;
	pvdata_t * kernelWeights = kernelPatch->data;
	pvdata_t kernel_ratio = 0.0f;
	pvdata_t hyper_ratio = 0.0f;
	float test_cond = 0.0f;
	for (int y = 0; y < ny; y++) {
		for (int k = 0; k < nk; k++) {
			kernel_ratio = kernelWeights[0] > 1.0E-10 ? kernelWeights[k]
					/ kernelWeights[0] : 0.0;
			hyper_ratio = hyperWeights[0] > 1.0E-10 ? hyperWeights[k]
					/ hyperWeights[0] : 0.0;
			test_cond = fabs(kernel_ratio + hyper_ratio) > 0 ? fabs(
					kernel_ratio - hyper_ratio) / fabs(kernel_ratio
					+ hyper_ratio) : 0.0f;
			if (test_cond > 0.01f) {
				const char * cHyper_filename = "gauss2d_hyper.txt";
				cHyPer->writeTextWeights(cHyper_filename, kPre);
				const char * cKernel_filename = "gauss2d_kernel.txt";
				cKernel->writeTextWeights(cKernel_filename, kPre);
			}
			assert(test_cond <= 0.001f);
		}
		// advance pointers in y
		hyperWeights += sy;
		kernelWeights += sy;
	}
	return status;
}

