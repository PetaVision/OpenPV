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
   InitRuleWeights(HyPerConn * conn);
   virtual ~InitRuleWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   InitRuleWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int ruleWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitRuleWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITRULEWEIGHTS_HPP_ */
