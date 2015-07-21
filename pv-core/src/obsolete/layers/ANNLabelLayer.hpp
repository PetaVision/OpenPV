/*
 * ANNLabelLayer.hpp
 *
 *  Created on: Jul. 23, 2013
 *      Author: xinhuazhang
 */

#ifndef ANNLABELLAYER_HPP_
#define ANNLABELLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNLabelLayer: public PV::ANNLayer {
public:
   ANNLabelLayer(const char * name, HyPerCol * hc);
   virtual ~ANNLabelLayer();
protected:
   ANNLabelLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);
private:
   int initialize_base();
};

} /* namespace PV */
#endif /* ANNLABELLAYER_HPP_ */
