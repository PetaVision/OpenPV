/*
 * InitSpreadOverArborsWeightsParams.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_
#define INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitSpreadOverArborsWeightsParams: public PV::InitWeightsParams {
public:
   InitSpreadOverArborsWeightsParams();
   InitSpreadOverArborsWeightsParams(HyPerConn * parentConn);
   virtual ~InitSpreadOverArborsWeightsParams();
   void calcOtherParams(int patchIndex);

   //get-set methods:
   inline float getInitWeight()        {return initWeight;}
   inline int getNumArbors()        {return numArbors;}


protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   int numArbors;
   float initWeight;

};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTSPARAMS_HPP_ */
