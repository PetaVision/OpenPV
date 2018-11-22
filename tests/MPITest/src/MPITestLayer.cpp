/*
 * MPITestLayer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "MPITestLayer.hpp"
#include <utils/conversions.h>

namespace PV {

MPITestLayer::MPITestLayer(const char *name, PVParams *params, Communicator *comm) : ANNLayer() {
   // MPITestLayer has no member variables to initialize in initialize_base()
   initialize(name, params, comm);
}

// set V to global x/y/f position
int MPITestLayer::setVtoGlobalPos() {
   auto const *layerGeometry = getComponentByType<LayerGeometry>();
   PVLayerLoc const *loc     = layerGeometry->getLayerLoc();
   float xScaleLog2          = layerGeometry->getXScale();
   float x0                  = xOriginGlobal(xScaleLog2);
   float dx                  = deltaX(xScaleLog2);
   for (int b = 0; b < loc->nbatch; b++) {
      for (int kLocal = 0; kLocal < getNumNeurons(); kLocal++) {
         int kGlobal        = globalIndexFromLocal(kLocal, *loc);
         int kxGlobal       = kxPos(kGlobal, loc->nxGlobal, loc->nyGlobal, loc->nf);
         float x_global_pos = (x0 + dx * kxGlobal);
         getV()[kLocal + b * getNumNeurons()] = x_global_pos;
      }
   }
   return PV_SUCCESS;
}

// set activity to global x/y/f position, using position in border/margin as required
int MPITestLayer::setActivitytoGlobalPos() {
   auto const *layerGeometry = getComponentByType<LayerGeometry>();
   PVLayerLoc const *loc     = layerGeometry->getLayerLoc();
   float xScaleLog2          = layerGeometry->getXScale();
   float x0                  = xOriginGlobal(xScaleLog2);
   float dx                  = deltaX(xScaleLog2);

   float *A = mActivityComponent->getComponentByType<ActivityBuffer>()->getReadWritePointer();
   for (int b = 0; b < loc->nbatch; b++) {
      for (int kLocalExt = 0; kLocalExt < getNumExtended(); kLocalExt++) {
         int kxLocalExt = kxPos(kLocalExt,
                                loc->nx + loc->halo.lt + loc->halo.rt,
                                loc->ny + loc->halo.dn + loc->halo.up,
                                loc->nf)
                          - loc->halo.lt;
         int kxGlobalExt    = kxLocalExt + loc->kx0;
         float x_global_pos = (x0 + dx * kxGlobalExt);
         int kyLocalExt     = kyPos(kLocalExt,
                                loc->nx + loc->halo.lt + loc->halo.rt,
                                loc->ny + loc->halo.dn + loc->halo.up,
                                loc->nf)
                          - loc->halo.up;
         int kyGlobalExt = kyLocalExt + loc->ky0;

         bool x_in_local_interior = kxLocalExt >= 0 && kxLocalExt < loc->nx;
         bool y_in_local_interior = kyLocalExt >= 0 && kyLocalExt < loc->ny;

         bool x_in_global_boundary = kxGlobalExt < 0 || kxGlobalExt >= loc->nxGlobal;
         bool y_in_global_boundary = kyGlobalExt < 0 || kyGlobalExt >= loc->nyGlobal;

         if ((x_in_global_boundary || x_in_local_interior)
             && (y_in_global_boundary || y_in_local_interior)) {
            A[kLocalExt + b * getNumExtended()] = x_global_pos;
         }
      }
   }
   return PV_SUCCESS;
}

void MPITestLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   ANNLayer::initialize(name, params, comm);
}

Response::Status MPITestLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
   if (Response::completed(status)) {
      setVtoGlobalPos();
      setActivitytoGlobalPos();
      status = Response::SUCCESS;
   }
   return status;
}

Response::Status MPITestLayer::checkUpdateState(double timed, double dt) {
   return Response::SUCCESS;
}

int MPITestLayer::publish(Communicator *comm, double timed) {
   setActivitytoGlobalPos();
   return HyPerLayer::publish(comm, timed);
}

} /* namespace PV */
