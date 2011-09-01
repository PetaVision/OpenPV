/*
 * KernelTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "KernelTestProbe.hpp"
#include "../PetaVision/src/io/StatsProbe.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

KernelTestProbe::KernelTestProbe(const char * filename, HyPerCol * hc, PVBufType buf_type, const char * msg)
: StatsProbe(filename, hc, buf_type, msg)
{
}

KernelTestProbe::KernelTestProbe(PVBufType buf_type, const char * msg)
: StatsProbe(buf_type, msg)
{
}


int KernelTestProbe::outputState(float time, HyPerLayer * l)
{
	int status = StatsProbe::outputState(time, l);
	assert((fMin>0.99)&&(fMin<1.010));
	assert((fMax>0.99)&&(fMax<1.010));
	assert((avg>0.99)&&(avg<1.010));

	return status;
}


} /* namespace PV */
