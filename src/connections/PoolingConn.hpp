/*
 * PoolingConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "components/ImpliedWeightsPair.hpp"
#include "components/PatchSize.hpp"
#include "connections/BaseConnection.hpp"

namespace PV {

class HyPerCol;

class PoolingConn : public BaseConnection {
  public:
   PoolingConn(char const *name, HyPerCol *hc);

   virtual ~PoolingConn();

   // get-methods for params
   int getPatchSizeX() const { return mPatchSize->getPatchSizeX(); }
   int getPatchSizeY() const { return mPatchSize->getPatchSizeY(); }
   int getPatchSizeF() const { return mPatchSize->getPatchSizeF(); }

   // other get-methods
   int getNumDataPatches() const { return mWeightsPair->getPreWeights()->getNumDataPatches(); }
   int getNumGeometryPatches() const {
      return mWeightsPair->getPreWeights()->getGeometry()->getNumPatches();
   }
   Patch const *getPatch(int kPre) { return &mWeightsPair->getPreWeights()->getPatch(kPre); }
   int getPatchStrideX() const { return mWeightsPair->getPreWeights()->getPatchStrideX(); }
   int getPatchStrideY() const { return mWeightsPair->getPreWeights()->getPatchStrideY(); }
   int getPatchStrideF() const { return mWeightsPair->getPreWeights()->getPatchStrideF(); }

  protected:
   PoolingConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual PatchSize *createPatchSize();

   virtual WeightsPairInterface *createWeightsPair();

  protected:
   PatchSize *mPatchSize              = nullptr;
   WeightsPairInterface *mWeightsPair = nullptr;

}; // class PoolingConn

} // namespace PV

#endif // POOLINGCONN_HPP_
