/*
 * LabelErrorLayer.hpp
 *
 *  Created on: Nov 30, 2013
 *      Author: garkenyon
 */

#ifndef LABELERRORLAYER_HPP_
#define LABELERRORLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class LabelErrorLayer: public PV::ANNLayer {
public:
   LabelErrorLayer(const char * name, HyPerCol * hc, int numChannels);
   LabelErrorLayer(const char * name, HyPerCol * hc);
   virtual ~LabelErrorLayer();
protected:
   LabelErrorLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
   int setParams(PVParams * params);
   int readErrScale(PVParams * params);
   int readIsBinary(PVParams * params);
private:
   int initialize_base();
   float errScale;
   int isBinary;
};

} /* namespace PV */
#endif /* LABELERRORLAYER_HPP_ */
