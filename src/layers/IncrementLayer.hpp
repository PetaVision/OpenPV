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
   int checkpointRead(const char * cpDir, double * timef);
   int checkpointWrite(const char * cpDir);
   virtual int updateState(double timef, double dt);

   inline pvdata_t * getVprev() {return Vprev;}

protected:
   IncrementLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   virtual int readVThreshParams(PVParams * params);
   /* static */ int updateState(double timef, double dt, bool * inited, double * next_update_time,
         double first_update_time, double display_period, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, pvdata_t * Vprev, int num_channels, pvdata_t * gSynHead, unsigned int * active_indices, unsigned int * num_active);

   virtual int setActivity();

private:
   int initialize_base();

protected:
   pvdata_t * Vprev;
   double displayPeriod;
   bool VInited;
   double firstUpdateTime; // necessary because of propagation delays
   double nextUpdateTime;
};

} /* namespace PV */
#endif /* INCREMENTLAYER_HPP_ */
