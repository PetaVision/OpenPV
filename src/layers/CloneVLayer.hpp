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

class CloneVLayer : public PV::HyPerLayer {
  public:
   CloneVLayer(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual bool activityIsSpiking() override { return false; }
   HyPerLayer *getOriginalLayer() { return mOriginalLayer; }
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual void createComponentTable(char const *description) override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual InternalStateBuffer *createInternalState() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer();
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status updateState(double timed, double dt) override;

  private:
   int initialize_base();

  protected:
   HyPerLayer *mOriginalLayer = nullptr;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
