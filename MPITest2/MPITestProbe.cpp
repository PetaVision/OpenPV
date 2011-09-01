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
MPITestProbe::MPITestProbe(const char * filename, HyPerCol * hc, PVBufType buf_type, const char * msg)
   : StatsProbe(filename, hc, buf_type, msg)
{
	cumAvg = 0.0f;
}

/**
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(PVBufType buf_type, const char * msg)
   : StatsProbe(buf_type, msg)
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
	double tol = 0.01f;
	fprintf(fp, "%s cum_time==%9.3f cumAvg==%f \n", msg, cum_time, (float) cumAvg);
	fflush(fp);
	if (time > 3.0f) {
		assert((fMin > (1.0f - tol)) && (fMin < (1.0f + tol)));
		assert((fMax > (1.0f - tol)) && (fMax < (1.0f + tol)));
		assert((avg > (1.0f - tol)) && (avg < (1.0f + tol)));
		assert((cumAvg > (cum_time - tol)) && (cumAvg < (cum_time + tol)));
	}

	return status;
}

}
