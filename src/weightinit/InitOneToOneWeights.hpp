/*
 * InitOneToOneWeights.hpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#ifndef INITONETOONEWEIGHTS_HPP_
#define INITONETOONEWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitOneToOneWeightsParams;

class InitOneToOneWeights: public PV::InitWeights {
public:
   InitOneToOneWeights();
   virtual ~InitOneToOneWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(int patchIndex);


protected:
   virtual int initialize_base();
   int createOneToOneConnection(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, float iWeight, InitWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTS_HPP_ */
