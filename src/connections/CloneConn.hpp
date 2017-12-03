/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#ifndef CLONECONN_HPP_
#define CLONECONN_HPP_

#include "connections/HyPerConn.hpp"

namespace PV {

class HyPerCol;

class CloneConn : public HyPerConn {
  public:
   CloneConn(char const *name, HyPerCol *hc);

   virtual ~CloneConn();

  protected:
   CloneConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual WeightsPair *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual NormalizeBase *createWeightNormalizer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;

   virtual int initializeState() override;

  protected:
}; // class CloneConn

} // namespace PV

#endif // CLONECONN_HPP_
