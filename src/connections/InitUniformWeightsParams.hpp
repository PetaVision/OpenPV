/*
 * InitUniformWeightsParams.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTSPARAMS_HPP_
#define INITUNIFORMWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitUniformWeightsParams: public PV::InitWeightsParams {
public:
   InitUniformWeightsParams();
   InitUniformWeightsParams(HyPerConn * parentConn);
   virtual ~InitUniformWeightsParams();

   //get-set methods:
   inline float getInitWeight()        {return iWeight;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float iWeight;

};

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTSPARAMS_HPP_ */
