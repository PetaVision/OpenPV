/*
 * InitSubUnitWeightsParams.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITSUBUNITWEIGHTSPARAMS_HPP_
#define INITSUBUNITWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitSubUnitWeightsParams: public PV::InitWeightsParams {
public:
   InitSubUnitWeightsParams();
   InitSubUnitWeightsParams(HyPerConn * parentConn);
   virtual ~InitSubUnitWeightsParams();
   void calcOtherParams(int patchIndex);


protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);

};

} /* namespace PV */
#endif /* INITSUBUNITWEIGHTSPARAMS_HPP_ */
