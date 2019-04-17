/*
 * WTAConn.hpp
 *
 *  Created on: Aug 15, 2018
 *      Author: pschultz
 */

#ifndef WTACONN_HPP_
#define WTACONN_HPP_

#include "BaseConnection.hpp"
#include "components/SingleArbor.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

/**
 * A connection class to perform winner-take-all computation.
 * If the pre-layer is nx-by-ny-by-nf, the post-layer must be nx-by-ny-by-1.
 * A single arbor is used, and the delay parameter is meaningful.
 * A WTADeliver component handles the update of the post-synaptic GSyn.
 */
class WTAConn : public BaseConnection {
  public:
   WTAConn(const char *name, PVParams *params, Communicator const *comm);

  protected:
   WTAConn();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual void fillComponentTable() override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual SingleArbor *createSingleArbor();
}; // class WTAConn

} // end of block for namespace PV

#endif /* WTACONN_HPP_ */
