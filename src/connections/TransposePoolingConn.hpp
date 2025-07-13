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
   TransposePoolingConn(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~TransposePoolingConn();

  protected:
   TransposePoolingConn();

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void fillComponentTable() override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual PatchSize *createPatchSize() override;

   virtual SharedWeights *createSharedWeights() override;

   virtual OriginalConnNameParam *createOriginalConnNameParam();

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class TransposePoolingConn

} // namespace PV

#endif // TRANSPOSEPOOLINGCONN_HPP_
