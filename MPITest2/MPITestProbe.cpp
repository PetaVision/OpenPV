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
MPITestProbe::MPITestProbe(const char * filename, HyPerCol * hc, const char * msg)
   : StatsProbe(filename, hc, msg)
{
	cumAvg = 0.0f;
}

/**
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(const char * msg)
   : StatsProbe(msg)
{
	cumAvg = 0.0f;
}


/**
 * @time
 * @l
 */
int MPITestProbe::outputState(float time, HyPerLayer * l) {
	int status = StatsProbe::outputState(time, l);
#ifdef PV_USE_MPI
	InterColComm * icComm = l->getParent()->icCommunicator();
	const int rcvProc = 0;
	if( icComm->commRank() != rcvProc ) {
		return status;
	}
#endif // PV_USE_MPI
	cumAvg += avg;
	double cum_time = time - 2.0f;
	double tol = 1e-5f;
	// the activity of each neuron should equal its global x/y/f position
	// minimum global postion is 0
	// maximum global position = nxGlobal
	int nf = l->getLayerLoc()->nf;
	int nxGlobal = l->getLayerLoc()->nxGlobal;
	int nyGlobal = l->getLayerLoc()->nyGlobal;
	int xScaleLog2 = l->getCLayer()->xScale;
	float min_global_xpos = xPosGlobal(0, xScaleLog2, nxGlobal, nyGlobal, nf);
	int kGlobal = nf * nxGlobal * nyGlobal - 1;
	float max_global_xpos = xPosGlobal(kGlobal, xScaleLog2, nxGlobal, nyGlobal, nf);
	float ave_global_xpos = (min_global_xpos + max_global_xpos) / 2.0f;
	fprintf(fp, "%s cum_time==%9.3f cumAvg==%f min_global_xpos==%f ave_global_xpos==%f max_global_xpos==%f \n",
			msg, cum_time, (float) cumAvg, min_global_xpos, ave_global_xpos, max_global_xpos);
	fflush(fp);
	if (time > 3.0f) {
		assert((fMin > (min_global_xpos - tol)) && (fMin < (min_global_xpos + tol)));
		assert((fMax > (max_global_xpos - tol)) && (fMax < (max_global_xpos + tol)));
		assert((avg > (ave_global_xpos - tol)) && (avg < (ave_global_xpos + tol)));
		assert((cumAvg > cum_time*(ave_global_xpos - tol)) && (cumAvg < cum_time*(ave_global_xpos + tol)));
	}

	return status;
}

}
