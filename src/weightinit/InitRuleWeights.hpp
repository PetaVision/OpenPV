/*
 * InitRuleWeights.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITRULEWEIGHTS_HPP_
#define INITRULEWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitRuleWeightsParams;

class InitRuleWeights: public PV::InitGauss2DWeights {
public:
   InitRuleWeights();
   virtual ~InitRuleWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   int initialize_base();

private:
   int ruleWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitRuleWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITRULEWEIGHTS_HPP_ */
