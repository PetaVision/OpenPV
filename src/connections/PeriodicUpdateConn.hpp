/*
 * PeriodicUpdateConn.hpp
 *
 *  Created on: May 18, 2011
 *      Author: peteschultz
 */

#ifdef OBSOLETE

#ifndef PERIODICUPDATECONN_HPP_
#define PERIODICUPDATECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class PeriodicUpdateConn: public PV::KernelConn {
public:
   PeriodicUpdateConn();
   PeriodicUpdateConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
   virtual ~PeriodicUpdateConn();

   float getWeightUpdatePeriod() {return weightUpdatePeriod;}
   float getNextWeightUpdate() {return nextWeightUpdate;}

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);

   float weightUpdatePeriod;
   float nextWeightUpdate;
};

}  // end namespace PV

#endif /* PERIODICUPDATECONN_HPP_ */
#endif // ifdef OBSOLETE
