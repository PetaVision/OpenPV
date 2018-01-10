/* TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *     Author: slundquist
 */

#ifndef TRANSPOSEPOOLINGCONN_HPP_
#define TRANSPOSEPOOLINGCONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/TransposeConn.hpp"

namespace PV {

class TransposePoolingConn : public TransposeConn {
  public:
   TransposePoolingConn(char const *name, HyPerCol *hc);

   virtual ~TransposePoolingConn();

  protected:
   TransposePoolingConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual BaseDelivery *createDeliveryObject() override;
   virtual SharedWeights *createSharedWeights() override;
   virtual WeightsPairInterface *createWeightsPair() override;
}; // class TransposePoolingConn

} // namespace PV

#endif // TRANSPOSEPOOLINGCONN_HPP_
