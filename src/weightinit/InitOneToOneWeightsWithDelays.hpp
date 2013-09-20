/*
 * InitOneToOneWeightsWithDelays.hpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#ifndef INITONETOONEWEIGHTSWITHDELAYS_HPP_
#define INITONETOONEWEIGHTSWITHDELAYS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitOneToOneWeightsWithDelaysParams;

class InitOneToOneWeightsWithDelays: public PV::InitWeights {
public:
   InitOneToOneWeightsWithDelays();
   virtual ~InitOneToOneWeightsWithDelays();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   void calcOtherParams(int patchIndex);


protected:
   virtual int initialize_base();
   int createOneToOneConnectionWithDelays(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, float iWeight, int nArbors, InitWeightsParams * weightParamPtr, int arborId);
};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSWITHDELAYS_HPP_ */
