/*
 * WTAConn.cpp
 *
 *  Created on: Aug 15, 2018
 *      Author: pschultz
 */

#include "WTAConn.hpp"
#include "columns/Factory.hpp"
#include "delivery/WTADelivery.hpp"

namespace PV {

WTAConn::WTAConn() {}

WTAConn::WTAConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void WTAConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseConnection::initialize(params, defaults, comm);
}

BaseDelivery *WTAConn::createDeliveryObject() {
   BaseObject *baseObject      = Factory::instance()->createByKeyword("WTADelivery", this);
   WTADelivery *deliveryObject = dynamic_cast<WTADelivery *>(baseObject);
   pvAssert(deliveryObject); // WTADelivery is a core keyword.
   return deliveryObject;
}

void WTAConn::fillComponentTable() {
   BaseConnection::fillComponentTable();
   auto *singleArbor = createSingleArbor();
   if (singleArbor) {
      addUniqueComponent(singleArbor);
   }
}

SingleArbor *WTAConn::createSingleArbor() {
   return new SingleArbor(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end of namespace PV block
