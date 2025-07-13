/*
 * PoolingIndexLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "PoolingIndexLayerInputBuffer.hpp"

namespace PV {

PoolingIndexLayerInputBuffer::PoolingIndexLayerInputBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PoolingIndexLayerInputBuffer::~PoolingIndexLayerInputBuffer() {}

void PoolingIndexLayerInputBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LayerInputBuffer::initialize(params, defaults, comm);
}

void PoolingIndexLayerInputBuffer::setObjectType() { mObjectType = "PoolingIndexLayerInputBuffer"; }

MPI_Op PoolingIndexLayerInputBuffer::setReductionOp() {
   FatalIf(
         !mDeliverySources.empty(),
         "PoolingIndexLayer \"%s\" cannot be a post-synaptic layer of any connection "
         "(only the postIndexLayer of a PoolingConn). Error in connection \"%s\".\n",
         getName(), mDeliverySources[0]->getName());
   mMPIReductionOp = MPI_MAX;
   return mMPIReductionOp;
}

void PoolingIndexLayerInputBuffer::resetGSynBuffers(double simulationTime, double deltaTime) {
   // Reset GSynBuffers does nothing, as the orig pooling connection deals with clearing this buffer
}

} // namespace PV
