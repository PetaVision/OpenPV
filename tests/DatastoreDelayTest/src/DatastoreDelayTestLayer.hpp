/*
 * DatastoreDelayTest.hpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#ifndef DATASTOREDELAYTESTLAYER_HPP_
#define DATASTOREDELAYTESTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class DatastoreDelayTestLayer : public HyPerLayer {

  public:
   DatastoreDelayTestLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~DatastoreDelayTestLayer();

  protected:
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual LayerInputBuffer *createLayerInput() override;

   virtual ActivityComponent *createActivityComponent() override;

}; // end class DatastoreDelayTestLayer

} // end namespace PV

#endif /* DATASTOREDELAYTESTLAYER_HPP_ */
