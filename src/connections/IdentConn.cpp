/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "columns/Factory.hpp"
#include "delivery/IdentDelivery.hpp"

namespace PV {

IdentConn::IdentConn() {}

IdentConn::IdentConn(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void IdentConn::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseConnection::initialize(name, params, comm);
}

BaseDelivery *IdentConn::createDeliveryObject() {
   BaseObject *baseObject        = Factory::instance()->createByKeyword("IdentDelivery", this);
   IdentDelivery *deliveryObject = dynamic_cast<IdentDelivery *>(baseObject);
   pvAssert(deliveryObject); // IdentDelivery is a core keyword.
   return deliveryObject;
}

void IdentConn::fillComponentTable() {
   BaseConnection::fillComponentTable();
   mSingleArbor = createSingleArbor();
   if (mSingleArbor) {
      addUniqueComponent(mSingleArbor);
   }
}

SingleArbor *IdentConn::createSingleArbor() {
   return new SingleArbor(name, parameters(), mCommunicator);
}

} // end of namespace PV block
