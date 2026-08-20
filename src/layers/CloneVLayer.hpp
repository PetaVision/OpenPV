/*
 * CloneVLayer.hpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#ifndef CLONEVLAYER_HPP_
#define CLONEVLAYER_HPP_

#include "BaseLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class CloneVLayer : public BaseLayer {
  public:
   CloneVLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void fillComponentTable() override;
   virtual LayerGeometry *createLayerGeometry() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();

  protected:
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
