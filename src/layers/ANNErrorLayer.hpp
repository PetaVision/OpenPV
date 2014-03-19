/*
 * ANNErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNERRORLAYER_HPP_
#define ANNERRORLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNErrorLayer: public PV::ANNLayer {
public:
   ANNErrorLayer(const char * name, HyPerCol * hc);
   virtual ~ANNErrorLayer();
protected:
   ANNErrorLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_errScale(enum ParamsIOFlag ioFlag);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
private:
   int initialize_base();
   float errScale;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
