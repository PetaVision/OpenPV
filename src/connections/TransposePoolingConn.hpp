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
   TransposePoolingConn(char const *name, HyPerCol *hc);

   virtual ~TransposePoolingConn();

  protected:
   TransposePoolingConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual PatchSize *createPatchSize() override;

   virtual OriginalConnNameParam *createOriginalConnNameParam();

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class TransposePoolingConn

} // namespace PV

#endif // TRANSPOSEPOOLINGCONN_HPP_
