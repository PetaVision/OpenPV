/*
 * PlasticConnTestLayer.cpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#include "PlasticConnTestLayer.hpp"
#include "../PetaVision/src/utils/conversions.h"

namespace PV {

PlasticConnTestLayer::PlasticConnTestLayer(const char * name, HyPerCol * hc, int numChannels) : ANNLayer(name, hc, numChannels) {
	initialize();
}

PlasticConnTestLayer::PlasticConnTestLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc, MAX_CHANNELS) {
	initialize();
}

// set V to global x/y/f position
int PlasticConnTestLayer::copyAtoV(){
	const PVLayerLoc * loc = getLayerLoc();
	pvdata_t * V = getV();
	pvdata_t * A = clayer->activity->data;
	for (int kLocal = 0; kLocal < getNumNeurons(); kLocal++){
		int kExtended = kIndexExtended(kLocal, loc->nx, loc->ny, loc->nf, loc->nb);
		V[kLocal] = A[kExtended];
	}
	return PV_SUCCESS;
}


// set activity to global x/y/f position, using position in border/margin as required
int PlasticConnTestLayer::setActivitytoGlobalPos(){
	for (int kLocalExt = 0; kLocalExt < getNumExtended(); kLocalExt++){
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


int PlasticConnTestLayer::initialize(){
	//int status = ANNLayer::initialize();  // parent class initialize already called in constructor (!!!violation of PV convention)
	setActivitytoGlobalPos();
	copyAtoV();

	return PV_SUCCESS;
}

int PlasticConnTestLayer::updateState(double timef, double dt)
{
   //updateV();
   //setActivity();
   //resetGSynBuffers();
   //updateActiveIndices();

   return PV_SUCCESS;
}

int PlasticConnTestLayer::publish(InterColComm* comm, double timef)
{
	setActivitytoGlobalPos();
	int status = comm->publish(this, clayer->activity);
	return status;

	//return HyPerLayer::publish(comm, time);
}



} /* namespace PV */
