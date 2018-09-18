/*
 * InputRegionLayer.hpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#ifndef INPUTREGIONLAYER_HPP_
#define INPUTREGIONLAYER_HPP_

#include "components/OriginalLayerNameParam.hpp"
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
   virtual bool needUpdate(double simTime, double dt) override;
   virtual bool activityIsSpiking() override { return false; }
   InputLayer *getOriginalLayer() { return mOriginalLayer; }

  protected:
   InputRegionLayer();
   int initialize(const char *name, HyPerCol *hc);
   void setObserverTable();
   virtual PhaseParam *createPhaseParam() override;
   virtual BoundaryConditions *createBoundaryConditions() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual InternalStateBuffer *createInternalState() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer();
   void checkLayerDimensions();
   virtual Response::Status allocateDataStructures() override;
   virtual int setActivity() override;

  private:
   int initialize_base();

  protected:
   InputLayer *mOriginalLayer = nullptr;
}; // class InputRegionLayer

} /* namespace PV */
#endif /* INPUTREGIONLAYER_HPP_ */
