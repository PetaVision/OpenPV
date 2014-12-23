/*
 * LCALayer.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: pschultz
 */

#ifndef LCALAYER_HPP_
#define LCALAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class LCALayer : public HyPerLayer {

// Methods
public:
   LCALayer(const char * name, HyPerCol * hc, int num_channels=MAX_CHANNELS);
   virtual ~LCALayer();

   const pvdata_t * getStimulus() {return stimulus;}
   float getThreshold() {return threshold;}
   float getThresholdSoftness() {return thresholdSoftness;}
   float getTimeConstantTau() {return timeConstantTau;}

   virtual int allocateDataStructures();

   virtual int updateState(double timef, double dt);
   virtual int checkpointWrite(const char * cpDir);

   // might not be exactly right as converting a constant input to an equivalent rate was calibrated for LIF
   // however, including the virtual function below
   // should help make response of LCA neuron approximately independent of time step size and allows
   // constant input to be interpreted as desired asymptotic value
   //
   //virtual float LCALayer::getChannelTimeConst(enum ChannelType channel_type){return timeConstantTau;};

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
   pvdata_t * stimulus;
   float threshold;
   float thresholdSoftness;
   float timeConstantTau;
}; // class LCALayer

} /* namespace PV */
#endif /* LCALAYER_HPP_ */
