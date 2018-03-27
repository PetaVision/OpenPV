/*
 * RescaleConn.hpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#ifndef RESCALECONN_HPP_
#define RESCALECONN_HPP_

#include "IdentConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class RescaleConn : public IdentConn {
  public:
   RescaleConn(const char *name, HyPerCol *hc);

  protected:
   RescaleConn();
   int initialize(const char *name, HyPerCol *hc);

   virtual BaseDelivery *createDeliveryObject() override;
}; // class RescaleConn

} // end of block for namespace PV

#endif /* RESCALECONN_HPP_ */
