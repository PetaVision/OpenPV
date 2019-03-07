/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#ifndef TRANSPOSEPOOLINGCONN_HPP_
#define TRANSPOSEPOOLINGCONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/PoolingConn.hpp"

namespace PV {

class TransposePoolingConn : public PoolingConn {
  public:
   TransposePoolingConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~TransposePoolingConn();

  protected:
   TransposePoolingConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void createComponentTable(char const *description) override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual PatchSize *createPatchSize() override;

   virtual OriginalConnNameParam *createOriginalConnNameParam();

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class TransposePoolingConn

} // namespace PV

#endif // TRANSPOSEPOOLINGCONN_HPP_
