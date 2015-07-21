/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef DISPARITYLCALAYER_HPP_
#define DISPARITYLCALAYER_HPP_

#include <layers/HyPerLCALayer.hpp>
#include <layers/Movie.hpp>

namespace PV {

class DisparityLCALayer: public PV::HyPerLCALayer{
public:
   DisparityLCALayer(const char * name, HyPerCol * hc);
   virtual ~DisparityLCALayer();
   virtual int communicateInitInfo();

protected:
   DisparityLCALayer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_disparityLayerName(enum ParamsIOFlag ioFlag);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);

private:
   int initialize_base();
   char * disparityLayerName;
   Movie* disparityLayer;


};

} /* namespace PV */
#endif /* HYPERLCALAYER_HPP_ */
