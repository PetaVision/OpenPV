/*
 * RescaleConn.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#include "RescaleConn.hpp"
#include "components/RescaleDelivery.hpp"

namespace PV {

RescaleConn::RescaleConn(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

RescaleConn::RescaleConn() { initialize_base(); }

int RescaleConn::initialize_base() { return PV_SUCCESS; }

int RescaleConn::initialize(char const *name, HyPerCol *hc) {
   return IdentConn::initialize(name, hc);
}

void RescaleConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("RescaleDelivery", name, parent);
   RescaleDelivery *deliveryObject = dynamic_cast<RescaleDelivery *>(baseObject);
   pvAssert(deliveryObject);
   setDeliveryObject(deliveryObject);
}

RescaleConn::~RescaleConn() {}

} /* namespace PV */
