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
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class PoolingIndexLayerInputBuffer : public LayerInputBuffer {
  public:
   PoolingIndexLayerInputBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~PoolingIndexLayerInputBuffer();

   float *getIndexBuffer(int b) { return &mBufferData[b * getBufferSize()]; }

  protected:
   PoolingIndexLayerInputBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);
   virtual void setObjectType() override;

   virtual void resetGSynBuffers(double simulationTime, double dt);

  protected:
};

} // namespace PV

#endif // POOLINGINDEXLAYERINPUTBUFFER_HPP_
