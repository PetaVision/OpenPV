/*
 * NaiveBayesLayer.hpp
 *
 *  Created on: Oct 29, 2012
 *      Author: garkenyon
 *
 *      Accumulates in-Class and out-Class counts and sums-of-activities.
 *      Activity stores log(prob_inClass/prob_outClass).
 *      CHANNEL_EXC stores bottom up activity
 *      CHANNEL_INH stores class mask
 */

#ifndef NAIVEBAYESLAYER_HPP_
#define NAIVEBAYESLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class NaiveBayesLayer: public PV::HyPerLayer {
public:
   NaiveBayesLayer(const char * name, HyPerCol * hc);
   virtual ~NaiveBayesLayer();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
protected:
   NaiveBayesLayer();
   int initialize(const char * name, HyPerCol * hc);
   /* static */
   int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, int columnID);
private:
   int initialize_base();
   long * inClassCount;
   long * outClassCount;
   double * inClassSum;
   double * outClassSum;
};

} // namespace PV
#endif /* NAIVEBAYESLAYER_HPP_ */
