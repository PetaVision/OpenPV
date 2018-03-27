/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "delivery/IdentDelivery.hpp"

namespace PV {

IdentConn::IdentConn() {}

IdentConn::IdentConn(const char *name, HyPerCol *hc) { initialize(name, hc); }

int IdentConn::initialize(const char *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

BaseDelivery *IdentConn::createDeliveryObject() {
   BaseObject *baseObject = Factory::instance()->createByKeyword("IdentDelivery", name, parent);
   IdentDelivery *deliveryObject = dynamic_cast<IdentDelivery *>(baseObject);
   pvAssert(deliveryObject);
   return deliveryObject;
}

void IdentConn::defineComponents() {
   BaseConnection::defineComponents();
   mSingleArbor = createSingleArbor();
   if (mSingleArbor) {
      addObserver(mSingleArbor);
   }
}

SingleArbor *IdentConn::createSingleArbor() { return new SingleArbor(name, parent); }

} // end of namespace PV block
