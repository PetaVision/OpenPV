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
class GapLayer: public CloneVLayer {
public:
   GapLayer(const char * name, HyPerCol * hc);
   virtual ~GapLayer();

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   virtual int updateState(double timef, double dt);

   // virtual int updateV();

protected:
   GapLayer();
   int initialize(const char * name, HyPerCol * hc);
      // use LIFGap as source layer instead (LIFGap updates gap junctions more accurately)
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ampSpikelet(enum ParamsIOFlag ioFlag);

   /* static */ int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * checkActive);
   virtual int setActivity();

private:
   int initialize_base();

   // Handled in CloneVLayer
   // char * sourceLayerName;
   // LIFGap * sourceLayer; // We don't call any LIFGap-specific methods so we can use originalLayer
   float ampSpikelet;

}; // class GapLayer

BaseObject * createGapLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* GAPLAYER_HPP_ */
