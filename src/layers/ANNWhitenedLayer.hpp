/*
 * ANNWhitenedLayer.hpp
 *
 *  Created on: Feb 15, 2013
 *      Author: garkenyon
 */

#ifndef ANNWHITENEDLAYER_HPP_
#define ANNWHITENEDLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNWhitenedLayer: public PV::ANNLayer {
public:
   ANNWhitenedLayer(const char * name, HyPerCol * hc, int numChannels);
   ANNWhitenedLayer(const char * name, HyPerCol * hc);
   virtual ~ANNWhitenedLayer();
protected:
   ANNWhitenedLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
private:
   int initialize_base();
};

} /* namespace PV */
#endif /* ANNWHITENEDLAYER_HPP_ */
