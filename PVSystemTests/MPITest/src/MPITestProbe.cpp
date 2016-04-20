/*
 * MPITestProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "MPITestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

/**
 * @probeName
 * @hc
 */
MPITestProbe::MPITestProbe(const char * probeName, HyPerCol * hc)
   : StatsProbe()
{
   initMPITestProbe(probeName, hc);
}

int MPITestProbe::initMPITestProbe_base() { return PV_SUCCESS; }

int MPITestProbe::initMPITestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

/**
 * @time
 * @l
 */
int MPITestProbe::outputState(double timed) {
	int status = StatsProbe::outputState(timed);
	InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
	const int rcvProc = 0;
	if( icComm->commRank() != rcvProc ) {
		return status;
	}
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
			getMessage(), min_global_xpos, ave_global_xpos, max_global_xpos);
	fflush(outputstream->fp);
   for(int b = 0; b < parent->getNBatch(); b++){
      if (timed > 3.0f) {
         assert((fMin[b]/min_global_xpos > (1 - tol)) && (fMin[b]/min_global_xpos < (1 + tol)));
         assert((fMax[b]/max_global_xpos > (1 - tol)) && (fMax[b]/max_global_xpos < (1 + tol)));
         assert((avg[b]/ave_global_xpos > (1 - tol)) && (avg[b]/ave_global_xpos < (1 + tol)));
      }
   }

	return status;
}

BaseObject * createMPITestProbe(char const * name, HyPerCol * hc) {
   return hc ? new MPITestProbe(name, hc) : NULL;
}

}
