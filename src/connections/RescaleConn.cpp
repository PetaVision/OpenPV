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

RescaleConn::RescaleConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void RescaleConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   IdentConn::initialize(params, defaults, comm);
}

BaseDelivery *RescaleConn::createDeliveryObject() {
   BaseObject *baseObject          = Factory::instance()->createByKeyword("RescaleDelivery", this);
   RescaleDelivery *deliveryObject = dynamic_cast<RescaleDelivery *>(baseObject);
   pvAssert(deliveryObject); // RescaleDelivery is a core keyword.
   return deliveryObject;
}

} // end of namespace PV block
