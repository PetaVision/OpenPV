/*
 * MPITestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "MPITestLayer.hpp"
#include "../PetaVision/src/utils/conversions.h"

namespace PV {

MPITestLayer::MPITestLayer(const char * name, HyPerCol * hc, int numChannels) : ANNLayer(name, hc, numChannels) {
	initialize();
}

MPITestLayer::MPITestLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc, MAX_CHANNELS) {
	initialize();
}

// set V to global x/y/f position
int MPITestLayer::setVtoGlobalPos(){
	for (int kLocal = 0; kLocal < clayer->numNeurons; kLocal++){
		int kGlobal = globalIndexFromLocal(kLocal, clayer->loc);
		int kxGlobal = kxPos(kGlobal, clayer->loc.nxGlobal, clayer->loc.nyGlobal, clayer->loc.nf);
		float xScaleLog2 = clayer->xScale;
		float x0 = xOriginGlobal(xScaleLog2);
		float dx = deltaX(xScaleLog2);
		float x_global_pos = (x0 + dx * kxGlobal);
		clayer->V[kLocal] = x_global_pos;
	}
	return PV_SUCCESS;
}


// set activity to global x/y/f position, using position in border/margin as required
int MPITestLayer::setActivitytoGlobalPos(){
	for (int kLocalExt = 0; kLocalExt < clayer->numExtended; kLocalExt++){
		int kxLocalExt = kxPos(kLocalExt, clayer->loc.nx + 2*clayer->loc.nb, clayer->loc.ny + 2*clayer->loc.nb, clayer->loc.nf) - clayer->loc.nb;
		int kxGlobalExt = kxLocalExt + clayer->loc.kx0;
		float xScaleLog2 = clayer->xScale;
		float x0 = xOriginGlobal(xScaleLog2);
		float dx = deltaX(xScaleLog2);
		float x_global_pos = (x0 + dx * kxGlobalExt);
		clayer->activity->data[kLocalExt] = x_global_pos;
	}
	return PV_SUCCESS;
}


int MPITestLayer::initialize(){
	int status = ANNLayer::initialize();
	setVtoGlobalPos();
	setActivitytoGlobalPos();

	return status;
}

int MPITestLayer::updateState(float time, float dt)
{
   //updateV();
   //setActivity();
   //resetGSynBuffers();
   //updateActiveIndices();

   return PV_SUCCESS;
}

int MPITestLayer::publish(InterColComm* comm, float time)
{
	setActivitytoGlobalPos();
	return HyPerLayer::publish(comm, time);
}



} /* namespace PV */
