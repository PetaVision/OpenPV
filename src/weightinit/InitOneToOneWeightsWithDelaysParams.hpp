/*
 * InitOneToOneWeightsWithDelaysParams.hpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 *
 */

#ifndef INITONETOONEWEIGHTSWITHDELAYSPARAMS_HPP_
#define INITONETOONEWEIGHTSWITHDELAYSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitOneToOneWeightsWithDelaysParams: public PV::InitWeightsParams {
public:
   InitOneToOneWeightsWithDelaysParams();
   InitOneToOneWeightsWithDelaysParams(HyPerConn * parentConn);
   virtual ~InitOneToOneWeightsWithDelaysParams();
   void calcOtherParams(int patchIndex);

   //get/set methods:
   inline float getInitWeight()        {return initWeight;}
   inline int getNumArbors()        {return numArbors;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);

private:
   float initWeight;
   int numArbors;

};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSWITHDELAYSPARAMS_HPP_ */
