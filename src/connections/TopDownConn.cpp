/*
 * TopDownConn.cpp
 *
 *  Created on: Oct 30, 2018
 *      Author: athresher
 */

#include "TopDownConn.hpp"
#include "delivery/TopDownDelivery.hpp"

namespace PV {

TopDownConn::TopDownConn() {}

TopDownConn::TopDownConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

int TopDownConn::initialize(const char *name, HyPerCol *hc) {
   int status = IdentConn::initialize(name, hc);
   return status;
}

BaseDelivery *TopDownConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("TopDownDelivery", name, parent);
   TopDownDelivery *deliveryObject = dynamic_cast<TopDownDelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

} // end of namespace PV block
