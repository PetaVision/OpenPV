/*
 * InitGaussianRandomWeightsParams.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_
#define INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_

#include "InitRandomWeightsParams.hpp"

namespace PV {

class InitGaussianRandomWeightsParams: public PV::InitRandomWeightsParams {
public:
   InitGaussianRandomWeightsParams();
   InitGaussianRandomWeightsParams(HyPerConn * parentConn);
   virtual ~InitGaussianRandomWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   //get-set methods:
   inline float getMean()        {return wGaussMean;}
   inline float getStDev()        {return wGaussStdev;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);
   void ioParam_wGaussMean(enum ParamsIOFlag ioFlag);
   void ioParam_wGaussStdev(enum ParamsIOFlag ioFlag);


private:
   float wGaussMean;
   float wGaussStdev;
};

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_ */
