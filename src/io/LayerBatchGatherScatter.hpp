#ifndef LAYERBATCHGATHERSCATTER_HPP_
#define LAYERBATCHGATHERSCATTER_HPP_

#include "include/PVLayerLoc.hpp"
#include "structures/Buffer.hpp"
#include "structures/MPIBlock.hpp"

#include <memory>

namespace PV {

class LayerBatchGatherScatter {

  public:
   LayerBatchGatherScatter(
         std::shared_ptr<MPIBlock const> mpiBlock,
         PVLayerLoc const &layerLoc,
         int rootProcessRank,
         bool localExtended,
         bool rootExtended);
   LayerBatchGatherScatter() {}
   ~LayerBatchGatherScatter() {}

   void gather(int mpiBatchIndex, float *rootDataLocation, float const *localDataLocation);
   void scatter(int mpiBatchIndex, float const *rootDataLocation, float *localDataLocation);
  
  private:
   void copyToDataLocation(float *dataLocation, Buffer<float> const &localDataBuffer);

  private:
   std::shared_ptr<MPIBlock const> mMPIBlock = nullptr;
   PVLayerLoc mLayerLoc;
   int mRootProcessRank = 0;
   bool mRootExtended = false;

}; // class LayerBatchGatherScatter

} // namespace PV

#endif // LAYERBATCHGATHERSCATTER_HPP_
