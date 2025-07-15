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

IdentConn::IdentConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void IdentConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   BaseConnection::initialize(paramsIO, comm);
   mWriteInitializeFromCheckpointFlag = false;
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
   return new SingleArbor(mParamsIO, mCommunicator);
}

} // end of namespace PV block
