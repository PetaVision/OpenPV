/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "BaseConnection.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class IdentConn : public BaseConnection {
  public:
   IdentConn(const char *name, HyPerCol *hc);

  protected:
   IdentConn();
   int initialize(const char *name, HyPerCol *hc);

   virtual ConnectionData *createConnectionData();

   virtual BaseDelivery *createDeliveryObject() override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // class IdentConn

} // end of block for namespace PV

#endif /* IDENTCONN_HPP_ */
