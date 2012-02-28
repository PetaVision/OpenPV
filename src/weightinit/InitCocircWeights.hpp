/*
 * InitCocircWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTS_HPP_
#define INITCOCIRCWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"
#include "InitCocircWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitCocircWeightsParams;

class InitCocircWeights: public PV::InitWeights {
public:
   InitCocircWeights();
//   InitCocircWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//         ChannelType channel);
   virtual ~InitCocircWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);


protected:
   virtual int initialize_base();
//   int initialize(const char * name, HyPerCol * hc,
//                  HyPerLayer * pre, HyPerLayer * post,
//                  ChannelType channel);

private:
   bool calcDistChordCocircKurvePreNKurvePost(
            float xDelta, float yDelta, int kfPost, InitCocircWeightsParams *weightParamPtr, float thPost);
   int cocircCalcWeights(pvdata_t * w_tmp, InitCocircWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTS_HPP_ */
