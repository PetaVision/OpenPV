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
   integrationTime = FLT_MAX;
   return PV_SUCCESS;
}

int LeakyIntegrator::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   return status;
}

int LeakyIntegrator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   return status;
}

void LeakyIntegrator::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "integrationTime", &integrationTime, integrationTime);
}

Response::Status LeakyIntegrator::updateState(double timed, double dt) {
   float *V          = getV();
   float const *gSyn = mLayerInput->getBufferData(0 /*batch index*/, CHANNEL_EXC);

   float decayfactor = std::exp(-(float)dt / integrationTime);
   for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
      V[k] *= decayfactor;
      V[k] += gSyn[k];
   }
   if (mLayerInput->getNumChannels() > 1) {
      float const *gSynInh = mLayerInput->getBufferData(0 /*batch index*/, CHANNEL_INH);
      for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
         V[k] -= gSynInh[k];
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
