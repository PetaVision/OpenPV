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

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float wMin;
   float wMax;
};

} /* namespace PV */
#endif /* INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_ */
