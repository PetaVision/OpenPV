/*
 * RescaleConn.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#include "RescaleConn.hpp"
#include "delivery/RescaleDelivery.hpp"

namespace PV {

RescaleConn::RescaleConn() {}

RescaleConn::RescaleConn(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

void RescaleConn::initialize(const char *name, PVParams *params, Communicator *comm) {
   IdentConn::initialize(name, params, comm);
}

BaseDelivery *RescaleConn::createDeliveryObject() {
   BaseObject *baseObject =
         Factory::instance()->createByKeyword("RescaleDelivery", name, parameters(), mCommunicator);
   RescaleDelivery *deliveryObject = dynamic_cast<RescaleDelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

} // end of namespace PV block
