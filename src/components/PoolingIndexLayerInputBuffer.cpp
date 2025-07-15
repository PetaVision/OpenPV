/*
 * PoolingIndexLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "PoolingIndexLayerInputBuffer.hpp"

namespace PV {

PoolingIndexLayerInputBuffer::PoolingIndexLayerInputBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

PoolingIndexLayerInputBuffer::~PoolingIndexLayerInputBuffer() {}

void PoolingIndexLayerInputBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   LayerInputBuffer::initialize(paramsIO, comm);
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
