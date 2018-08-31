/*
 * WTAConn.cpp
 *
 *  Created on: Aug 15, 2018
 *      Author: pschultz
 */

#include "WTAConn.hpp"
#include "delivery/WTADelivery.hpp"

namespace PV {

WTAConn::WTAConn() {}

WTAConn::WTAConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

int WTAConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

BaseDelivery *WTAConn::createDeliveryObject() {
   BaseObject *baseObject      = Factory::instance()->createByKeyword("WTADelivery", name, parent);
   WTADelivery *deliveryObject = dynamic_cast<WTADelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

void WTAConn::setObserverTable() {
   BaseConnection::setObserverTable();
   auto *singleArbor = createSingleArbor();
   if (singleArbor) {
      addUniqueComponent(singleArbor->getDescription(), singleArbor);
   }
}

SingleArbor *WTAConn::createSingleArbor() { return new SingleArbor(name, parent); }

} // end of namespace PV block
