/*
 * InitDistributedWeights.hpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 */

#ifndef INITDISTRIBUTEDWEIGHTS_HPP_
#define INITDISTRIBUTEDWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitDistributedWeightsParams.hpp"

namespace PV {

class InitDistributedWeights: public PV::InitWeights {
public:
   InitDistributedWeights();
//   InitUniformRandomWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//         ChannelType channel);
   virtual ~InitDistributedWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   virtual int initialize_base();
//   int initialize(const char * name, HyPerCol * hc,
//                  HyPerLayer * pre, HyPerLayer * post,
//                  ChannelType channel);

private:
   int distributedWeights(/* PVPatch * wp */ pvdata_t * dataStart, float minwgt, float maxwgt, InitDistributedWeightsParams *weightParamPtr);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
