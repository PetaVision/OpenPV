/*
 * InputRegionLayer.hpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#ifndef INPUTREGIONLAYER_HPP_
#define INPUTREGIONLAYER_HPP_

#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

namespace PV {

/**
 * A class whose activity buffer has a nonzero value whereever an associated
 * InputLayer's activity buffer is occupied by pixels from the image (as opposed
 * to padding created by offsetting or resizing), and zero otherwise
 */
class InputRegionLayer : public HyPerLayer {
  public:
   InputRegionLayer(const char *name, HyPerCol *hc);
   virtual ~InputRegionLayer();
   virtual int requireChannel(int channelNeeded, int *numChannelsResult) override;
   virtual bool needUpdate(double timestamp, double dt) override;
   virtual bool activityIsSpiking() override { return false; }
   InputLayer *getOriginalLayer() { return originalLayer; }

  protected:
   InputRegionLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_phase(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_valueBC(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer(HyPerLayer *layer);
   void checkLayerDimensions();
   virtual Response::Status allocateDataStructures() override;
   virtual void allocateGSyn() override;
   virtual void allocateV() override;
   virtual void allocateActivity() override;
   virtual int setActivity() override;

  private:
   int initialize_base();

  protected:
   char *originalLayerName;
   InputLayer *originalLayer;
}; // class InputRegionLayer

} /* namespace PV */
#endif /* INPUTREGIONLAYER_HPP_ */
