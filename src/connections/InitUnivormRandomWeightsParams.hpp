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

class InitUnivormRandomWeightsParams: public PV::InitWeightsParams {
public:
   InitUnivormRandomWeightsParams();
   InitUnivormRandomWeightsParams(HyPerConn * parentConn);
   virtual ~InitUnivormRandomWeightsParams();

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
