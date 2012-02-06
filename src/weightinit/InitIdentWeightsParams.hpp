/*
 * InitIdentWeightsParams.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITIDENTWEIGHTSPARAMS_HPP_
#define INITIDENTWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitIdentWeightsParams: public PV::InitWeightsParams {
public:
   InitIdentWeightsParams();
   InitIdentWeightsParams(HyPerConn * parentConn);
   virtual ~InitIdentWeightsParams();
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);
};

} /* namespace PV */
#endif /* INITIDENTWEIGHTSPARAMS_HPP_ */
