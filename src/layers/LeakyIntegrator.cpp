/*
 * LeakyIntegrator.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#include "LeakyIntegrator.hpp"
#include <cmath>

namespace PV {

LeakyIntegrator::LeakyIntegrator(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

LeakyIntegrator::LeakyIntegrator() { initialize_base(); }

int LeakyIntegrator::initialize_base() {
   numChannels     = 1;
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

int LeakyIntegrator::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   assert(numChannels == 1);
   return status;
}

int LeakyIntegrator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyIntegrator::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "integrationTime", &integrationTime, integrationTime);
}

Response::Status LeakyIntegrator::updateState(double timed, double dt) {
   float *V    = getV();
   float *gSyn = GSyn[0];

   float decayfactor = std::exp(-(float)dt / integrationTime);
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      V[k] *= decayfactor;
      V[k] += GSyn[0][k];
      if (numChannels > 1) {
         V[k] -= GSyn[1][k];
      }
   }
   int nx     = getLayerLoc()->nx;
   int ny     = getLayerLoc()->ny;
   int nf     = getLayerLoc()->nf;
   int nbatch = getLayerLoc()->nbatch;

   PVHalo const *halo = &getLayerLoc()->halo;
   float *A           = getActivity();
   setActivity_PtwiseLinearTransferLayer(
         nbatch,
         getNumNeurons(),
         A,
         V,
         nx,
         ny,
         nf,
         halo->lt,
         halo->rt,
         halo->dn,
         halo->up,
         numVertices,
         verticesV,
         verticesA,
         slopes);
   return Response::SUCCESS;
}

LeakyIntegrator::~LeakyIntegrator() {}

} /* namespace PV */
