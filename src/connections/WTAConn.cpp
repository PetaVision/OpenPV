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

WTAConn::WTAConn(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void WTAConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseConnection::initialize(name, params, comm);
}

BaseDelivery *WTAConn::createDeliveryObject() {
   BaseObject *baseObject      = createSubobject("WTADelivery");
   WTADelivery *deliveryObject = dynamic_cast<WTADelivery *>(baseObject);
   pvAssert(deliveryObject); // WTADelivery is a core keyword.
   return deliveryObject;
}

void WTAConn::createComponentTable(char const *description) {
   BaseConnection::createComponentTable(description);
   auto *singleArbor = createSingleArbor();
   if (singleArbor) {
      addUniqueComponent(singleArbor->getDescription(), singleArbor);
   }
}

SingleArbor *WTAConn::createSingleArbor() {
   return new SingleArbor(name, parameters(), mCommunicator);
}

} // end of namespace PV block
