/*
 * InitUniformRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMRANDOMWEIGHTS_HPP_
#define INITUNIFORMRANDOMWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

class InitUniformRandomWeights: public PV::InitWeights {
public:
   InitUniformRandomWeights();
//   InitUniformRandomWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//         ChannelType channel);
   virtual ~InitUniformRandomWeights();

   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   virtual int initialize_base();
//   int initialize(const char * name, HyPerCol * hc,
//                  HyPerLayer * pre, HyPerLayer * post,
//                  ChannelType channel);

private:
   int uniformWeights(/* PVPatch * wp */ pvdata_t * dataStart, float minwgt, float maxwgt, InitUniformRandomWeightsParams *weightParamPtr);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
