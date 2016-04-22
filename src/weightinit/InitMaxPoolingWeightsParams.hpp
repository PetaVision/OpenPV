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
   InitMaxPoolingWeightsParams(const char * name, HyPerCol * hc);
   virtual ~InitMaxPoolingWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

protected:
   virtual int initialize_base();
   int initialize(const char * name, HyPerCol * hc);


};

} /* namespace PV */
#endif /* INITMAXPOOLINGWEIGHTSPARAMS_HPP_ */
