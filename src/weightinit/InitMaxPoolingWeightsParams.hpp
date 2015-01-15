/*
 * InitMaxPoolingWeightsParams.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITMAXPOOLINGWEIGHTSPARAMS_HPP_
#define INITMAXPOOLINGWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitMaxPoolingWeightsParams: public PV::InitWeightsParams {
public:
   InitMaxPoolingWeightsParams();
   InitMaxPoolingWeightsParams(HyPerConn * parentConn);
   virtual ~InitMaxPoolingWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


};

} /* namespace PV */
#endif /* INITMAXPOOLINGWEIGHTSPARAMS_HPP_ */
