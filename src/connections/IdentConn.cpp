/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "components/NoCheckpointConnectionData.hpp"
#include "delivery/IdentDelivery.hpp"

namespace PV {

IdentConn::IdentConn() {}

IdentConn::IdentConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

int IdentConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

BaseDelivery *IdentConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("IdentDelivery", name, parent);
   IdentDelivery *deliveryObject = dynamic_cast<IdentDelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

ConnectionData *IdentConn::createConnectionData() {
   return new NoCheckpointConnectionData(name, parent);
}

int IdentConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseConnection::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   const PVLayerLoc *preLoc  = getPre()->getLayerLoc();
   const PVLayerLoc *postLoc = getPost()->getLayerLoc();
   if (preLoc->nx != postLoc->nx) {
      ErrorLog().printf(
            "%s requires pre and post nx be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nx,
            postLoc->nx);
      status = PV_FAILURE;
   }
   if (preLoc->ny != postLoc->ny) {
      ErrorLog().printf(
            "%s requires pre and post ny be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->ny,
            postLoc->ny);
      status = PV_FAILURE;
   }
   if (preLoc->nf != postLoc->nf) {
      ErrorLog().printf(
            "%s requires pre and post nf be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nf,
            postLoc->nf);
      status = PV_FAILURE;
   }
   if (preLoc->nbatch != postLoc->nbatch) {
      ErrorLog().printf(
            "%s requires pre and post nbatch be equal (%d versus %d).\n",
            getDescription_c(),
            preLoc->nbatch,
            postLoc->nbatch);
      status = PV_FAILURE;
   }
   FatalIf(
         status != PV_SUCCESS,
         "IdentConn \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: "
         "%dx%dx%d vs. %dx%dx%d\n",
         name,
         getPre()->getName(),
         getPost()->getName(),
         preLoc->nx,
         preLoc->ny,
         preLoc->nf,
         postLoc->nx,
         postLoc->ny,
         postLoc->nf);
   return status;
}

} // end of namespace PV block
