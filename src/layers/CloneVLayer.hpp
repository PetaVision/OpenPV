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
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int requireChannel(int channelNeeded, int *numChannelsResult) override;
   virtual int allocateGSyn() override;
   virtual int
   requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis) override;
   virtual bool activityIsSpiking() override { return false; }
   HyPerLayer *getOriginalLayer() { return originalLayer; }
   virtual ~CloneVLayer();

  protected:
   CloneVLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;
   virtual int allocateV() override;
   virtual int registerData(Checkpointer *checkpointer) override;
   virtual int initializeV() override;
   virtual int readVFromCheckpoint(Checkpointer *checkpointer) override;
   virtual int updateState(double timed, double dt) override;

  private:
   int initialize_base();

  protected:
   char *originalLayerName;
   HyPerLayer *originalLayer;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
