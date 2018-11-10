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
   CloneVLayer(const char *name, HyPerCol *hc);
   virtual bool activityIsSpiking() override { return false; }
   HyPerLayer *getOriginalLayer() { return mOriginalLayer; }
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual void createComponentTable(char const *description) override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();

  private:
   int initialize_base();

  protected:
   HyPerLayer *mOriginalLayer = nullptr;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
