/*
 * GenerativeLayer.hpp
 *
 * A class derived from ANNLayer where the update rule is
 * dAnew = excitatorychannel - inhibitorychannel + auxChannelCoeff*auxiliarychannel - d(log(1+old^2))/d(old)
 * dAnew = persistence*dAold + (1-persistenceOfMemory)*dAnew
 * A = A + relaxation*dAnew
 * dAold = dAnew
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVELAYER_HPP_
#define GENERATIVELAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class GenerativeLayer : public ANNLayer {
public:
   GenerativeLayer(const char * name, HyPerCol * hc);
//   GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type);
   ~GenerativeLayer();

   virtual int updateState(float timef, float dt);

   pvdata_t getRelaxation() {return relaxation;}
   pvdata_t getActivityThreshold() { return activityThreshold; }

protected:
   GenerativeLayer();
   int initialize(const char * name, HyPerCol * hc);
   /* static */ int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence, pvdata_t activity_threshold, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   static pvdata_t reduce_relaxation(int num_neurons, pvdata_t * V, pvdata_t * dV, pvdata_t relaxation);

   pvdata_t relaxation; // V(new) = V(old) - relaxation*(gradient wrt V)
   pvdata_t activityThreshold;  // values with absolute value below threshold are zero
   pvdata_t auxChannelCoeff; // coefficient on channel 2 in update rule
   pvdata_t sparsityTermCoeff; // coefficient in front of the sparsity function in energy
   pvdata_t persistence;  // "stickiness" of the rate of change of weights
   pvdata_t * dV;  // buffer holding past rate of change of weights
   pvdata_t * sparsitytermderivative;  // buffer holding derivative of sparsity function
private:
   int initialize_base();
};

}  // end namespace PV block

#endif /* GENERATIVELAYER_HPP_ */
