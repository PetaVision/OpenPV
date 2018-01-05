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

   virtual ConnectionData *createConnectionData() override;

   virtual BaseDelivery *createDeliveryObject() override;
}; // class IdentConn

} // end of block for namespace PV

#endif /* IDENTCONN_HPP_ */
