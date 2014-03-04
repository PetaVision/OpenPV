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

class CloneVLayer: public PV::HyPerLayer {
public:
   CloneVLayer(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int allocateGSyn();
   virtual int initializeState();
   virtual int requireMarginWidth(int marginWidthNeeded, int * marginWidthResult);
   virtual ~CloneVLayer();

protected:
   CloneVLayer();
   int initialize(const char * name, HyPerCol * hc);
   int setParams(PVParams * params);
   void readOriginalLayerName(PVParams * params);
   int allocateV();
   virtual int checkpointRead(const char * cpDir, double * timed);
   virtual int checkpointWrite(const char * cpDir);
   virtual int doUpdateState(double timed, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);

private:
   int initialize_base();

protected:
   char * originalLayerName;
   HyPerLayer * originalLayer;
};

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
