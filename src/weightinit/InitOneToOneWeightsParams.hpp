/*
 * InitOneToOneWeightsParams.hpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 *
 *      Note: this InitWeights class accepts patch sizes greater than 1, but
 *      it doesn't make sense to do that because it will connect all other
 *      points except the presynaptic neuron directly under the postsynaptic
 *      neuron to 0.
 */

#ifndef INITONETOONEWEIGHTSPARAMS_HPP_
#define INITONETOONEWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitOneToOneWeightsParams: public PV::InitWeightsParams {
public:
   InitOneToOneWeightsParams();
   InitOneToOneWeightsParams(HyPerConn * parentConn);
   virtual ~InitOneToOneWeightsParams();
   void calcOtherParams(int patchIndex);

   //get/set methods:
   inline float getInitWeight()        {return initWeight;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);

private:
   float initWeight;

};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSPARAMS_HPP_ */
