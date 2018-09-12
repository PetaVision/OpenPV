/*
 * SpikingIntegrator.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "SpikingIntegrator.hpp"
#include <cmath>

namespace PV {

SpikingIntegrator::SpikingIntegrator(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

SpikingIntegrator::SpikingIntegrator() { initialize_base(); }

int SpikingIntegrator::initialize_base() {
   numChannels     = 1;
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

int SpikingIntegrator::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels == 1);
   return status;
}

int SpikingIntegrator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_Vthresh(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void SpikingIntegrator::ioParam_Vthresh(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "Vthresh", &Vthresh, Vthresh);
}

void SpikingIntegrator::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "integrationTime", &integrationTime, integrationTime);
}

Response::Status SpikingIntegrator::updateState(double timed, double dt) {
   float *V    = getV();
   float *gSyn = GSyn[0];

   float *A           = getActivity();
   float decayfactor = std::exp(-(float)dt / integrationTime);
   int nb     = getLayerLoc()->nbatch;
   int nx     = getLayerLoc()->nx;
   int ny     = getLayerLoc()->ny;
   int nf     = getLayerLoc()->nf;
   int lt     = getLayerLoc()->halo.lt;
   int rt     = getLayerLoc()->halo.rt;
   int dn     = getLayerLoc()->halo.dn;
   int up     = getLayerLoc()->halo.up;
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      V[k] *= decayfactor;
      V[k] += GSyn[0][k]*(1-decayfactor);
      if (numChannels > 1) {
         V[k] -= GSyn[1][k]*(1-decayfactor);
      }
      int kExt = kIndexExtendedBatch(k,nb,nx,ny,nf,lt,rt,dn,up);
      if (V[k] > Vthresh) {
         V[k] = 0;
         A[kExt] = 1;
      }
      else {
         A[kExt] = 0;
      }
   }

   return Response::SUCCESS;
}

SpikingIntegrator::~SpikingIntegrator() {}

} /* namespace PV */
