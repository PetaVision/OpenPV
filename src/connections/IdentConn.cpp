/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "components/IdentDelivery.hpp"

namespace PV {

IdentConn::IdentConn() { initialize_base(); }

IdentConn::IdentConn(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
} // end of IdentConn::initialize_base()

int IdentConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

#ifdef PV_USE_CUDA
void IdentConn::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // Never receive from gpu
   receiveGpu = false;
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "receiveGpu", false /*correctValue*/);
   }
}
#endif // PV_USE_CUDA

void IdentConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
}

void IdentConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      numAxonalArborLists = 1;
      parent->parameters()->handleUnnecessaryParameter(
            name, "numAxonalArbors", numAxonalArborLists);
   }
}

void IdentConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

int IdentConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseConnection::communicateInitInfo(message);
   assert(pre && post);
   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();
   if (preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "IdentConn \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: "
               "%dx%dx%d vs. %dx%dx%d\n",
               name,
               preLayerName,
               postLayerName,
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               postLoc->nx,
               postLoc->ny,
               postLoc->nf);
      }
      exit(EXIT_FAILURE);
   }
   return status;
}

void IdentConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("IdentDelivery", name, parent);
   IdentDelivery *deliveryObject = dynamic_cast<IdentDelivery *>(baseObject);
   pvAssert(deliveryObject);
   setDeliveryObject(deliveryObject);
}

int IdentConn::deliver() {
   getDeliveryObject()->deliver(nullptr);
   return PV_SUCCESS;
}

void IdentConn::deliverUnitInput(float *recvBuffer) {
   getDeliveryObject()->deliverUnitInput(nullptr, recvBuffer);
}

} // end of namespace PV block
