/*
 * MPITestActivityBuffer.cpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#include "MPITestActivityBuffer.hpp"

namespace PV {

MPITestActivityBuffer::MPITestActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

MPITestActivityBuffer::~MPITestActivityBuffer() {}

void MPITestActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void MPITestActivityBuffer::setObjectType() { mObjectType = "MPITestActivityBuffer"; }

Response::Status
MPITestActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   setActivityToGlobalPos();
   return Response::SUCCESS;
}

void MPITestActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   setActivityToGlobalPos();
}

// set activity to global x/y/f position, using position in border/margin as required
void MPITestActivityBuffer::setActivityToGlobalPos() {
   PVLayerLoc const *loc = mLayerGeometry->getLayerLoc();
   float xScaleLog2      = mLayerGeometry->getXScale();
   float x0              = xOriginGlobal(xScaleLog2);
   float dx              = deltaX(xScaleLog2);

   float *A = mBufferData.data();
   for (int b = 0; b < loc->nbatch; b++) {
      for (int kLocalExt = 0; kLocalExt < getBufferSize(); kLocalExt++) {
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
            A[kLocalExt + b * getBufferSize()] = x_global_pos;
         }
      }
   }
}

} // namespace PV
