/*
 * InitOneToOneWeights.hpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#ifndef INITONETOONEWEIGHTS_HPP_
#define INITONETOONEWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

// TODO make InitOneToOneWeights a derived class of InitUniformWeights
class InitOneToOneWeights : public InitWeights {
  protected:
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);

  public:
   InitOneToOneWeights(char const *name, HyPerCol *hc);
   virtual ~InitOneToOneWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void calcWeights(int patchIndex, int arborId) override;
   void calcOtherParams(int patchIndex);

  protected:
   InitOneToOneWeights();
   int initialize(char const *name, HyPerCol *hc);
   int createOneToOneConnection(float *dataStart, int patchIndex, float weightInit);

  protected:
   float mWeightInit;
}; // class InitOneToOneWeights

} /* namespace PV */
#endif /* INITONETOONEWEIGHTS_HPP_ */
