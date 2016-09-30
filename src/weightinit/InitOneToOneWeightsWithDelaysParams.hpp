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

// TODO make InitOneToOneWeightsWithDelaysParams a derived class of InitOneToOneWeightsParams
class InitOneToOneWeightsWithDelaysParams : public PV::InitWeightsParams {
  public:
   InitOneToOneWeightsWithDelaysParams();
   InitOneToOneWeightsWithDelaysParams(const char *name, HyPerCol *hc);
   virtual ~InitOneToOneWeightsWithDelaysParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);

   // get/set methods:
   inline float getInitWeight() { return initWeight; }

  protected:
   virtual int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);

  private:
   float initWeight;
};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSWITHDELAYSPARAMS_HPP_ */
