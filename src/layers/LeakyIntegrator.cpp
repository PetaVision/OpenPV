/*
 * LeakyIntegrator.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegrator.hpp"

namespace PV {

LeakyIntegrator::LeakyIntegrator(const char* name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

LeakyIntegrator::LeakyIntegrator() {
   initialize_base();
}

int LeakyIntegrator::initialize_base() {
   numChannels = 1;
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

int LeakyIntegrator::initialize(const char * name, HyPerCol * hc) {
   PVParams * params = hc->parameters();
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels==1);
   return status;
}

int LeakyIntegrator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyIntegrator::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "integrationTime", &integrationTime, integrationTime);
}

int LeakyIntegrator::updateState(double timed, double dt) {
   pvdata_t * V = getV();
   pvdata_t * gSyn = GSyn[0];
   pvdata_t decayfactor = (pvdata_t) exp(-dt/integrationTime);
   for (int k=0; k<getNumNeurons(); k++) {
      V[k] *= decayfactor;
      V[k] += gSyn[k];
   }
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nf = getLayerLoc()->nf;
   int nb = getLayerLoc()->nb;
   pvdata_t * A = getActivity();
   int status = setActivity_HyPerLayer(getNumNeurons(), A, V, nx, ny, nf, nb);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(getNumNeurons(), V, AMin, VThresh, AShift, VWidth, A, nx, ny, nf, nb);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(getNumNeurons(), V, AMax, A, nx, ny, nf, nb);
   return status;
}

LeakyIntegrator::~LeakyIntegrator() {
}

} /* namespace PV */
