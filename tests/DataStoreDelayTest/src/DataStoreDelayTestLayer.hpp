/*
 * DataStoreDelayTest.hpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#ifndef DATASTOREDELAYTESTLAYER_HPP_
#define DATASTOREDELAYTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class DataStoreDelayTestLayer : public HyPerLayer {

  public:
   DataStoreDelayTestLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~DataStoreDelayTestLayer();

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;

}; // end class DataStoreDelayTestLayer

} // end namespace PV

#endif /* DATASTOREDELAYTESTLAYER_HPP_ */
