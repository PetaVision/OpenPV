/*
 * HyPerConn.hpp
 *
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "components/ConnectionData.hpp"
#include "components/WeightsPair.hpp"
//#include "weightupdaters/WeightUpdater.hpp"
#include "connections/BaseConnection.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "weightinit/InitWeights.hpp"

namespace PV {

class HyPerCol;

class HyPerConn : public BaseConnection {
  public:
   HyPerConn(char const *name, HyPerCol *hc);

   virtual ~HyPerConn();

   // get-methods for params
   int getPatchSizeX() const { return mWeightsPair->getPatchSizeX(); }
   int getPatchSizeY() const { return mWeightsPair->getPatchSizeY(); }
   int getPatchSizeF() const { return mWeightsPair->getPatchSizeF(); }
   int getSharedWeights() const { return mWeightsPair->getSharedWeights(); }

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

   int getPatchStrideX() const { return mWeightsPair->getPreWeights()->getPatchStrideX(); }
   int getPatchStrideY() const { return mWeightsPair->getPreWeights()->getPatchStrideY(); }
   int getPatchStrideF() const { return mWeightsPair->getPreWeights()->getPatchStrideF(); }

   int calcDataIndexFromPatchIndex(int patchIndex) {
      return mWeightsPair->getPreWeights()->calcDataIndexFromPatchIndex(patchIndex);
   }

  protected:
   HyPerConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual WeightsPair *createWeightsPair();
   virtual InitWeights *createWeightInitializer();
   virtual NormalizeBase *createWeightNormalizer();
   virtual BaseDelivery *createDeliveryObject() override;
   // virtual WeightUpdater *createWeightUpdater() override;

   virtual int initializeState() override;

  protected:
   WeightsPair *mWeightsPair        = nullptr;
   InitWeights *mWeightInitializer  = nullptr;
   NormalizeBase *mWeightNormalizer = nullptr;

}; // class HyPerConn

} // namespace PV

#endif // HYPERCONN_HPP_
