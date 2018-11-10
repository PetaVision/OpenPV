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
   DatastoreDelayTestLayer(const char *name, HyPerCol *hc);
   virtual ~DatastoreDelayTestLayer();

  protected:
   int initialize(const char *name, HyPerCol *hc);

   virtual LayerInputBuffer *createLayerInput();

   virtual ActivityComponent *createActivityComponent() override;

}; // end class DatastoreDelayTestLayer

} // end namespace PV

#endif /* DATASTOREDELAYTESTLAYER_HPP_ */
