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

RescaleConn::RescaleConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

int RescaleConn::initialize(const char *name, HyPerCol *hc) {
   int status = IdentConn::initialize(name, hc);
   return status;
}

BaseDelivery *RescaleConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("RescaleDelivery", name, parent);
   RescaleDelivery *deliveryObject = dynamic_cast<RescaleDelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

} // end of namespace PV block
