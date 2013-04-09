/*
 * InitUnivormRandomWeightsParams.hpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_
#define INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitUniformRandomWeightsParams: public PV::InitWeightsParams {
public:
   InitUniformRandomWeightsParams();
   InitUniformRandomWeightsParams(HyPerConn * parentConn);
   virtual ~InitUniformRandomWeightsParams();

   //get-set methods:
   inline float getWMin()        {return wMin;}
   inline float getWMax()        {return wMax;}
   inline float getSparseFraction()        {return sparseFraction;}

protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float wMin;
   float wMax;
   float sparseFraction;  // fraction of weights identically zero:  0 (default) -> no sparseness, 1 -> all weights == 0
};

} /* namespace PV */
#endif /* INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_ */
