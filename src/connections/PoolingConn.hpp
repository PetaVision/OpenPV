/*
 * PoolingConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "components/ArborList.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "components/NoCheckpointConnectionData.hpp"
#include "connections/BaseConnection.hpp"

namespace PV {

class HyPerCol;

class PoolingConn : public BaseConnection {
  public:
   PoolingConn(char const *name, HyPerCol *hc);

   virtual ~PoolingConn();

   // get-methods for params
   int getPatchSizeX() const { return mWeightsPair->getPatchSizeX(); }
   int getPatchSizeY() const { return mWeightsPair->getPatchSizeY(); }
   int getPatchSizeF() const { return mWeightsPair->getPatchSizeF(); }

   int getNumAxonalArbors() const { return mArborList->getNumAxonalArbors(); }
   int getDelay(int arbor) const { return mArborList->getDelay(arbor); }

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

   virtual ConnectionData *createConnectionData() override;

   virtual ArborList *createArborList();

   virtual BaseDelivery *createDeliveryObject() override;

   virtual ImpliedWeightsPair *createWeightsPair();

  protected:
   ArborList *mArborList            = nullptr;
   ImpliedWeightsPair *mWeightsPair = nullptr;

}; // class PoolingConn

} // namespace PV

#endif // POOLINGCONN_HPP_
