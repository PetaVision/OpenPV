/*
 * HyPerConn.hpp
 *
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "components/ArborList.hpp"
#include "components/ConnectionData.hpp"
#include "components/SharedWeights.hpp"
#include "components/WeightsPair.hpp"
#include "connections/BaseConnection.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "weightinit/InitWeights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HyPerCol;

class HyPerConn : public BaseConnection {
  public:
   HyPerConn(char const *name, HyPerCol *hc);

   virtual ~HyPerConn();

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   // get-methods for params
   int getPatchSizeX() const { return mPatchSize->getPatchSizeX(); }
   int getPatchSizeY() const { return mPatchSize->getPatchSizeY(); }
   int getPatchSizeF() const { return mPatchSize->getPatchSizeF(); }
   int getSharedWeights() const { return mSharedWeights->getSharedWeights(); }

   int getNumAxonalArbors() const { return mArborList->getNumAxonalArbors(); }
   int getDelay(int arbor) const { return mArborList->getDelay(arbor); }

   int getStrength() const { return mWeightNormalizer->getStrength(); }

   // other get-methods
   int getNumDataPatches() const { return mWeightsPair->getPreWeights()->getNumDataPatches(); }
   int getNumGeometryPatches() const {
      return mWeightsPair->getPreWeights()->getGeometry()->getNumPatches();
   }
   Patch const *getPatch(int kPre) { return &mWeightsPair->getPreWeights()->getPatch(kPre); }
   float *getWeightsDataStart(int arbor) const {
      return mWeightsPair->getPreWeights()->getData(arbor);
   }
   float *getWeightsDataHead(int arbor, int dataIndex) const {
      return mWeightsPair->getPreWeights()->getDataFromDataIndex(arbor, dataIndex);
   }
   float *getWeightsData(int arbor, int patchIndex) {
      auto *preWeights = mWeightsPair->getPreWeights();
      return preWeights->getDataFromPatchIndex(arbor, patchIndex)
             + preWeights->getPatch(patchIndex).offset;
   }
   float const *getDeltaWeightsDataStart(int arbor) const;
   float const *getDeltaWeightsDataHead(int arbor, int dataIndex) const;
   int getPatchStrideX() const { return mWeightsPair->getPreWeights()->getPatchStrideX(); }
   int getPatchStrideY() const { return mWeightsPair->getPreWeights()->getPatchStrideY(); }
   int getPatchStrideF() const { return mWeightsPair->getPreWeights()->getPatchStrideF(); }

   double getLastUpdateTime() const { return mWeightsPair->getPreWeights()->getTimestamp(); }

   int calcDataIndexFromPatchIndex(int patchIndex) {
      return mWeightsPair->getPreWeights()->calcDataIndexFromPatchIndex(patchIndex);
   }

  protected:
   HyPerConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual BaseDelivery *createDeliveryObject() override;
   virtual ArborList *createArborList();
   virtual PatchSize *createPatchSize();
   virtual SharedWeights *createSharedWeights();
   virtual WeightsPairInterface *createWeightsPair();
   virtual InitWeights *createWeightInitializer();
   virtual NormalizeBase *createWeightNormalizer();
   virtual BaseWeightUpdater *createWeightUpdater();

   Response::Status respondConnectionUpdate(std::shared_ptr<ConnectionUpdateMessage const> message);

   Response::Status
   respondConnectionNormalize(std::shared_ptr<ConnectionNormalizeMessage const> message);

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual Response::Status initializeState() override;

  protected:
   ArborList *mArborList              = nullptr;
   PatchSize *mPatchSize              = nullptr;
   SharedWeights *mSharedWeights      = nullptr;
   WeightsPairInterface *mWeightsPair = nullptr;
   InitWeights *mWeightInitializer    = nullptr;
   NormalizeBase *mWeightNormalizer   = nullptr;
   BaseWeightUpdater *mWeightUpdater  = nullptr;

   Timer *mUpdateTimer = nullptr;

}; // class HyPerConn

} // namespace PV

#endif // HYPERCONN_HPP_
