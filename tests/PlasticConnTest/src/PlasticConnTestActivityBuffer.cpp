/*
 * PlasticConnTestActivityBuffer.cpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#include "PlasticConnTestActivityBuffer.hpp"

namespace PV {

PlasticConnTestActivityBuffer::PlasticConnTestActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

PlasticConnTestActivityBuffer::~PlasticConnTestActivityBuffer() {}

void PlasticConnTestActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityBuffer::initialize(paramsIO, comm);
}

void PlasticConnTestActivityBuffer::setObjectType() {
   mObjectType = "PlasticConnTestActivityBuffer";
}

Response::Status PlasticConnTestActivityBuffer::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   setActivityToGlobalPos();
   return Response::SUCCESS;
}

void PlasticConnTestActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   setActivityToGlobalPos();
}

// set activity to global x/y/f position, using position in border/margin as required
void PlasticConnTestActivityBuffer::setActivityToGlobalPos() {
   PVLayerLoc const *loc = mLayerGeometry->getLayerLoc();
   float xScaleLog2      = mLayerGeometry->getXScale();
   float x0              = xOriginGlobal(xScaleLog2);
   float dx              = deltaX(xScaleLog2);
   float *A              = mBufferData.data();
   for (int kLocalExt = 0; kLocalExt < getBufferSize(); kLocalExt++) {
      int kxLocalExt = kxPos(kLocalExt,
                             loc->nx + loc->halo.lt + loc->halo.rt,
                             loc->ny + loc->halo.dn + loc->halo.up,
                             loc->nf)
                       - loc->halo.lt;
      int kxGlobalExt    = kxLocalExt + loc->kx0;
      float x_global_pos = (x0 + dx * kxGlobalExt);
      A[kLocalExt]       = x_global_pos;
   }
}

} // namespace PV
