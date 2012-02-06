/*
 * InitTransposeWeightsParams.hpp
 *
 *  Created on: Aug 15, 2011
 *      Author: kpeterson
 */

#ifndef INITTRANSPOSEWEIGHTSPARAMS_HPP_
#define INITTRANSPOSEWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitTransposeWeightsParams: public PV::InitWeightsParams {
public:
   InitTransposeWeightsParams();
   InitTransposeWeightsParams(HyPerConn * parentConn);
   virtual ~InitTransposeWeightsParams();
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);
};

} /* namespace PV */
#endif /* INITTRANSPOSEWEIGHTSPARAMS_HPP_ */
