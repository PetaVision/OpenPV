/*
 * InitSpreadOverArborsWeightsParams.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_
#define INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

class InitSpreadOverArborsWeightsParams: public PV::InitGauss2DWeightsParams {
public:
   InitSpreadOverArborsWeightsParams();
   InitSpreadOverArborsWeightsParams(char const * name, HyPerCol * hc);
   InitSpreadOverArborsWeightsParams(HyPerConn * parentConn);
   virtual ~InitSpreadOverArborsWeightsParams();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);

   //get-set methods:
   inline float getInitWeight()        {return initWeight;}


protected:
   int initialize_base();
   int initialize(char const * name, HyPerCol * hc);
   void ioParam_weightInit(enum ParamsIOFlag ioFlag);

private:
   float initWeight;

};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_ */
