/*
 * TopDownConn.hpp
 *
 *  Created on: Oct 30, 2018
 *      Author: athresher
 */

#ifndef TOPDOWNCONN_HPP_
#define TOPDOWNCONN_HPP_

#include "IdentConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class TopDownConn : public IdentConn {
  public:
   TopDownConn(const char *name, HyPerCol *hc);

  protected:
   TopDownConn();
   int initialize(const char *name, HyPerCol *hc);

   virtual BaseDelivery *createDeliveryObject() override;
}; // class TopDownConn

} // end of block for namespace PV

#endif /* TOPDOWNCONN_HPP_ */
