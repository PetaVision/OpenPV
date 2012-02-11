/*
 * IncrementLayer.hpp
 *
 *  Created on: Feb 7, 2012
 *      Author: pschultz
 *
 *
 */

#ifndef INCREMENTLAYER_HPP_
#define INCREMENTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class IncrementLayer: public PV::ANNLayer {
public:
   IncrementLayer(const char * name, HyPerCol * hc, int numChannels=MAX_CHANNELS);
   virtual ~IncrementLayer();

protected:
   IncrementLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   virtual int readVThreshParams(PVParams * params);
   virtual int updateState(float timef, float dt);
   virtual int setActivity();

private:
   int initialize_base();

protected:
   pvdata_t * Vprev;
   float displayPeriod;
   bool VInited;
   float firstUpdateTime; // necessary because of propagation delays
   float nextUpdateTime;
};

} /* namespace PV */
#endif /* INCREMENTLAYER_HPP_ */
