/*
 * LCALayer.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: pschultz
 */

#ifndef LCALAYER_HPP_
#define LCALAYER_HPP_

#include <src/layers/HyPerLayer.hpp>

namespace PV {

class LCALayer : public HyPerLayer {

// Methods
public:
   LCALayer(const char * name, HyPerCol * hc, int num_channels=MAX_CHANNELS);
   virtual ~LCALayer();
   virtual int updateState(float timef, float dt);
   float getThreshold() {return threshold;}
   float getThresholdSoftness() {return thresholdSoftness;}
   float getTimeConstantTau() {return timeConstantTau;}

protected:
   LCALayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels);

   virtual float readThreshold() {return getParent()->parameters()->value(name, "threshold", 1.0);}
   virtual float readThresholdSoftness() {return getParent()->parameters()->value(name, "thresholdSoftness", 0.0);}
   virtual float readTimeConstantTau() {return getParent()->parameters()->value(name, "timeConstantTau", 10.0);}

private:
   int initialize_base();

// Member variables
protected:
   float threshold;
   float thresholdSoftness;
   float timeConstantTau;
}; // class LCALayer

} /* namespace PV */
#endif /* LCALAYER_HPP_ */
