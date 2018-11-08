/*
 * PlasticConnTestLayer.cpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#include "PlasticConnTestLayer.hpp"
#include <utils/conversions.h>

namespace PV {

PlasticConnTestLayer::PlasticConnTestLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

// set V to global x/y/f position
int PlasticConnTestLayer::copyAtoV() {
   const PVLayerLoc *loc = getLayerLoc();
   float *V              = getV();
   float const *A        = mActivityComponent->getActivity();
   for (int kLocal = 0; kLocal < getNumNeurons(); kLocal++) {
      int kExtended = kIndexExtended(
            kLocal,
            loc->nx,
            loc->ny,
            loc->nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up);
      V[kLocal] = A[kExtended];
   }
   return PV_SUCCESS;
}

// set activity to global x/y/f position, using position in border/margin as required
int PlasticConnTestLayer::setActivitytoGlobalPos() {
   auto *layerGeometry   = getComponentByType<LayerGeometry>();
   PVLayerLoc const *loc = layerGeometry->getLayerLoc();
   float xScaleLog2      = layerGeometry->getXScale();
   float x0              = xOriginGlobal(xScaleLog2);
   float dx              = deltaX(xScaleLog2);
   auto *activityBuffer  = mActivityComponent->getComponentByType<ActivityBuffer>();
   pvAssert(activityBuffer);
   float *A = activityBuffer->getReadWritePointer();
   for (int kLocalExt = 0; kLocalExt < getNumExtended(); kLocalExt++) {
      int kxLocalExt = kxPos(kLocalExt,
                             loc->nx + loc->halo.lt + loc->halo.rt,
                             loc->ny + loc->halo.dn + loc->halo.up,
                             loc->nf)
                       - loc->halo.lt;
      int kxGlobalExt    = kxLocalExt + loc->kx0;
      float x_global_pos = (x0 + dx * kxGlobalExt);
      A[kLocalExt]       = x_global_pos;
   }
   return PV_SUCCESS;
}

void PlasticConnTestLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   ANNLayer::initialize(name, params, comm);
}

Response::Status PlasticConnTestLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   if (Response::completed(status)) {
      setActivitytoGlobalPos();
      copyAtoV();
      status = Response::SUCCESS;
   }
   return status;
}

Response::Status PlasticConnTestLayer::updateState(double timef, double dt) {
   return Response::SUCCESS;
}

int PlasticConnTestLayer::publish(Communicator *comm, double timef) {
   setActivitytoGlobalPos();
   return HyPerLayer::publish(comm, timef);
}

} /* namespace PV */
