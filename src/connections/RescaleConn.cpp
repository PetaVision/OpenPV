/*
 * RescaleConn.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#include "RescaleConn.hpp"
#include "columns/Factory.hpp"
#include "delivery/RescaleDelivery.hpp"

namespace PV {

RescaleConn::RescaleConn() {}

RescaleConn::RescaleConn(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void RescaleConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   IdentConn::initialize(name, params, comm);
}

BaseDelivery *RescaleConn::createDeliveryObject() {
   BaseObject *baseObject          = Factory::instance()->createByKeyword("RescaleDelivery", this);
   RescaleDelivery *deliveryObject = dynamic_cast<RescaleDelivery *>(baseObject);
   pvAssert(deliveryObject); // RescaleDelivery is a core keyword.
   return deliveryObject;
}

} // end of namespace PV block
