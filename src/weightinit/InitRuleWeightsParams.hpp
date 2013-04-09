/*
 * InitRuleWeightsParams.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITRULEWEIGHTSPARAMS_HPP_
#define INITRULEWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

class InitRuleWeightsParams: public PV::InitGauss2DWeightsParams {
public:
   InitRuleWeightsParams();
   InitRuleWeightsParams(HyPerConn * parentConn);
   virtual ~InitRuleWeightsParams();
   void calcOtherParams(int patchIndex);

   //get/set methods:
   inline float getStrength()        {return strength;}

protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);

private:
   float strength;

};

} /* namespace PV */
#endif /* INITRULEWEIGHTSPARAMS_HPP_ */
