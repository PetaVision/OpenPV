/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef POOLINGINDEXLAYER_HPP_
#define POOLINGINDEXLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class PoolingIndexLayer : public HyPerLayer {
  public:
   PoolingIndexLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~PoolingIndexLayer();

  protected:
   PoolingIndexLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent();
}; // end of class PoolingIndexLayer

} // end namespace PV

#endif /* ANNLAYER_HPP_ */
