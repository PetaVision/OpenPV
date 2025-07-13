/*
 * PoolingIndexLayerInputBuffer.hpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#ifndef POOLINGINDEXLAYERINPUTBUFFER_HPP_
#define POOLINGINDEXLAYERINPUTBUFFER_HPP_

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A LayerInputBuffer type for receiving the locations of the maxima when maxpooling is with
 * needPostIndexLayer is true.
 */
class PoolingIndexLayerInputBuffer : public LayerInputBuffer {
  public:
   PoolingIndexLayerInputBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~PoolingIndexLayerInputBuffer();

   virtual void recvAllSynapticInput(double simTime, double deltaTime) override {}

   float *getIndexBuffer(int b) { return &mBufferData[b * getBufferSize()]; }

  protected:
   PoolingIndexLayerInputBuffer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual void setObjectType() override;

   virtual MPI_Op setReductionOp() override;

   virtual void resetGSynBuffers(double simulationTime, double dt) override;

  protected:
};

} // namespace PV

#endif // POOLINGINDEXLAYERINPUTBUFFER_HPP_
