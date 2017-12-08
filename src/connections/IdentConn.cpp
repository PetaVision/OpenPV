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
   int status                = BaseConnection::communicateInitInfo(message);
   const PVLayerLoc *preLoc  = getPre()->getLayerLoc();
   const PVLayerLoc *postLoc = getPost()->getLayerLoc();
   if (preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
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
      }
      exit(EXIT_FAILURE);
   }
   return status;
}

} // end of namespace PV block
