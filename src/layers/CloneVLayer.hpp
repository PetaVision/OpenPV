/*
 * CloneVLayer.hpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#ifndef CLONEVLAYER_HPP_
#define CLONEVLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class CloneVLayer : public HyPerLayer {
  public:
   CloneVLayer(const char *name, PVParams *params, Communicator const *comm);
   HyPerLayer *getOriginalLayer() { return mOriginalLayer; }
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();

  protected:
   HyPerLayer *mOriginalLayer = nullptr;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
