/*
 * StatsProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "MPITestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

/**
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}


/**
 * @time
 * @l
 */
int MPITestProbe::outputState(double timed) {
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
	InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
	const int rcvProc = 0;
	if( icComm->commRank() != rcvProc ) {
		return status;
	}
#endif // PV_USE_MPI
	double tol = 1e-4f;

	// if many to one connection, each neuron should receive its global x/y/f position
	// if one to many connection, the position of the nearest sending cell is received
	// assume sending layer has scale factor == 1
	int xScaleLog2 = getTargetLayer()->getCLayer()->xScale;

	// determine min/max position of receiving layer
	const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
	int nf = loc->nf;
	int nxGlobal = loc->nxGlobal;
	int nyGlobal = loc->nyGlobal;
	float min_global_xpos = xPosGlobal(0, xScaleLog2, nxGlobal, nyGlobal, nf);
	int kGlobal = nf * nxGlobal * nyGlobal - 1;
	float max_global_xpos = xPosGlobal(kGlobal, xScaleLog2, nxGlobal, nyGlobal, nf);

	if (xScaleLog2 < 0) {
		float xpos_shift = 0.5 - min_global_xpos;
		min_global_xpos = 0.5;
		max_global_xpos -= xpos_shift;
	}
	float ave_global_xpos = (min_global_xpos + max_global_xpos) / 2.0f;

	fprintf(outputstream->fp, "%s min_global_xpos==%f ave_global_xpos==%f max_global_xpos==%f \n",
			msg, min_global_xpos, ave_global_xpos, max_global_xpos);
	fflush(outputstream->fp);
	if (timed > 3.0f) {
		assert((fMin/min_global_xpos > (1 - tol)) && (fMin/min_global_xpos < (1 + tol)));
		assert((fMax/max_global_xpos > (1 - tol)) && (fMax/max_global_xpos < (1 + tol)));
		assert((avg/ave_global_xpos > (1 - tol)) && (avg/ave_global_xpos < (1 + tol)));
	}

	return status;
}

}
