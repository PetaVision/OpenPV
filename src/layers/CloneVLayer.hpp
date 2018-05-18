/*
 * CloneVLayer.hpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#ifndef CLONEVLAYER_HPP_
#define CLONEVLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class CloneVLayer : public PV::HyPerLayer {
  public:
   CloneVLayer(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual int requireChannel(int channelNeeded, int *numChannelsResult) override;
   virtual void allocateGSyn() override;
   virtual bool activityIsSpiking() override { return false; }
   HyPerLayer *getOriginalLayer() { return originalLayer; }
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;
   virtual void allocateV() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual void initializeV() override;
   virtual void readVFromCheckpoint(Checkpointer *checkpointer) override;
   virtual Response::Status updateState(double timed, double dt) override;

  private:
   int initialize_base();

  protected:
   char *originalLayerName;
   HyPerLayer *originalLayer;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
