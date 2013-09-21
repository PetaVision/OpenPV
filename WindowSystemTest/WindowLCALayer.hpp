/*
 * WindowLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef WINDOWLCALAYER_HPP_
#define WINDOWLCALAYER_HPP_

#include <layers/HyPerLCALayer.hpp>

namespace PV {

class WindowLCALayer: public PV::HyPerLCALayer{
public:
   WindowLCALayer(const char * name, HyPerCol * hc, int numChannels);
   WindowLCALayer(const char * name, HyPerCol * hc);
   virtual ~WindowLCALayer();
protected:
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
};

}
#endif /* WINDOWLCALAYER_HPP_*/
