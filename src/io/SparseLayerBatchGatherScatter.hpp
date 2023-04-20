#ifndef SPARSELAYERBATCHGATHERSCATTER_HPP_
#define SPARSELAYERBATCHGATHERSCATTER_HPP_

#include "include/PVLayerLoc.h"
#include "structures/MPIBlock.hpp"
#include "structures/SparseList.hpp"

#include <memory>

namespace PV {

class SparseLayerBatchGatherScatter {

  public:
   SparseLayerBatchGatherScatter(
         std::shared_ptr<MPIBlock const> mpiBlock,
         PVLayerLoc const &layerLoc,
         int rootProcessRank,
         bool localExtended,
         bool rootExtended);
   SparseLayerBatchGatherScatter() {}
   ~SparseLayerBatchGatherScatter() {}

   void gather(
         int mpiBatchIndex,
         SparseList<float> *rootSparseList,
         SparseList<float> const *localSparseList);
   void scatter(
         int mpiBatchIndex,
         SparseList<float> const *rootSparseList,
         SparseList<float> *localSparseList);

  private:
   std::shared_ptr<MPIBlock const> mMPIBlock = nullptr;
   PVLayerLoc mLayerLoc;
   int mRootProcessRank = 0;
   bool mRootExtended = false;

}; // class SparseLayerBatchGatherScatter

} // namespace PV

#endif // SPARSELAYERBATCHGATHERSCATTER_HPP_
