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
   virtual bool needUpdate(double simTime, double dt) const override;
   virtual bool activityIsSpiking() override { return false; }

  protected:
   InputRegionLayer();
   int initialize(const char *name, HyPerCol *hc);
   void createComponentTable(char const *description);
   virtual PhaseParam *createPhaseParam() override;
   virtual BoundaryConditions *createBoundaryConditions() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag) override;
}; // class InputRegionLayer

} /* namespace PV */
#endif /* INPUTREGIONLAYER_HPP_ */
