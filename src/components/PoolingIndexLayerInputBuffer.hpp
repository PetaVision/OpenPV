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
   PoolingIndexLayerInputBuffer(char const *name, HyPerCol *hc);

   virtual ~PoolingIndexLayerInputBuffer();

   float *getIndexBuffer(int b) { return &mBufferData[b * getBufferSize()]; }

  protected:
   PoolingIndexLayerInputBuffer() {}

   int initialize(char const *name, HyPerCol *hc);
   virtual void setObjectType() override;

   virtual void resetGSynBuffers(double simulationTime, double dt);

  protected:
};

} // namespace PV

#endif // POOLINGINDEXLAYERINPUTBUFFER_HPP_
