/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "BaseConnection.hpp"
#include "components/SingleArbor.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class IdentConn : public BaseConnection {
  public:
   IdentConn(const char *name, PVParams *params, Communicator *comm);

  protected:
   IdentConn();
   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual void createComponentTable(char const *description) override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual SingleArbor *createSingleArbor();

  protected:
   ArborList *mSingleArbor = nullptr;
}; // class IdentConn

} // end of block for namespace PV

#endif /* IDENTCONN_HPP_ */
