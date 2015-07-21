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

// TODO make InitOneToOneWeightsParams a derived class of InitUniformWeightsParams
class InitOneToOneWeightsParams: public PV::InitWeightsParams {
public:
   InitOneToOneWeightsParams();
   InitOneToOneWeightsParams(const char * name, HyPerCol * hc);
   virtual ~InitOneToOneWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);

   //get/set methods:
   inline float getInitWeight()        {return initWeight;}

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);

private:
   float initWeight;

};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSPARAMS_HPP_ */
