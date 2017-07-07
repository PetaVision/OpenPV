/*
 * GapLayer.hpp
 * can be used to implement gap junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef GAPLAYER_HPP_
#define GAPLAYER_HPP_

#include "CloneVLayer.hpp"
#include "LIFGap.hpp"

namespace PV {

// CloneLayer can be used to implement gap junctions between spiking neurons
class GapLayer : public CloneVLayer {
  public:
   GapLayer(const char *name, HyPerCol *hc);
   virtual ~GapLayer();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;

   virtual int updateState(double timef, double dt) override;

  protected:
   GapLayer();
   int initialize(const char *name, HyPerCol *hc);
   // use LIFGap as source layer instead (LIFGap updates gap junctions more accurately)
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_ampSpikelet(enum ParamsIOFlag ioFlag);

   /* static */ int updateState(
         double timef,
         double dt,
         const PVLayerLoc *loc,
         float *A,
         float *V,
         float *checkActive);
   virtual int setActivity() override;

  private:
   int initialize_base();

   // Handled in CloneVLayer
   float ampSpikelet;

}; // class GapLayer

} // namespace PV

#endif /* GAPLAYER_HPP_ */
