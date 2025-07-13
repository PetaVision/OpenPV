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
   PoolingIndexLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~PoolingIndexLayer();

  protected:
   PoolingIndexLayer();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
}; // end of class PoolingIndexLayer

} // end namespace PV

#endif /* ANNLAYER_HPP_ */
