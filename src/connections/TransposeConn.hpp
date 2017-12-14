/* TransposeConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#ifndef TRANSPOSECONN_HPP_
#define TRANSPOSECONN_HPP_

#include "connections/HyPerConn.hpp"

namespace PV {

class TransposeConn : public HyPerConn {
  public:
   TransposeConn(char const *name, HyPerCol *hc);

   virtual ~TransposeConn();

  protected:
   TransposeConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual WeightsPair *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual NormalizeBase *createWeightNormalizer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;

   virtual int initializeState() override;

  protected:
}; // class TransposeConn

} // namespace PV

#endif // TRANSPOSECONN_HPP_
