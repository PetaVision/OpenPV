/*
 * InitGaussianRandomWeightsParams.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_
#define INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitGaussianRandomWeightsParams: public PV::InitWeightsParams {
public:
   InitGaussianRandomWeightsParams();
   InitGaussianRandomWeightsParams(HyPerConn * parentConn);
   virtual ~InitGaussianRandomWeightsParams();

   //get-set methods:
   inline float getMean()        {return wGaussMean;}
   inline float getStDev()        {return wGaussStdev;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float wGaussMean;
   float wGaussStdev;
};

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTSPARAMS_HPP_ */
