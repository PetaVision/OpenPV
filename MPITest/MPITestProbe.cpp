/*
 * StatsProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "MPITestProbe.hpp"
#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <float.h>      // FLT_MAX/MIN
#include <string.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(const char * filename, HyPerCol * hc, PVBufType buf_type, const char * msg)
   : StatsProbe(filename, hc, buf_type, msg)
{
	cumSum = 0.0f;
	cumAvg = 0.0f;
}

/**
 * @type
 * @msg
 */
MPITestProbe::MPITestProbe(PVBufType buf_type, const char * msg)
   : StatsProbe(buf_type, msg)
{
	cumSum = 0.0f;
	cumAvg = 0.0f;
}


/**
 * @time
 * @l
 */
int MPITestProbe::outputState(float time, HyPerLayer * l)
{
	int status = StatsProbe::outputState(time, l);

	fprintf(fp, "%s t==%9.3f cumSum==%f cumAvg==%f \n", msg, time,
              (float) cumSum, (float) cumAvg);
	fflush(fp);

	return 0;
}

}
