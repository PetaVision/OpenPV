/*
 * InitGaborWeights.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTS_HPP_
#define INITGABORWEIGHTS_HPP_

#include <weightinit/InitGauss2DWeights.hpp>

namespace PV {

class InitGaborWeights : public PV::InitGauss2DWeights {
  protected:
   virtual void ioParam_lambda(enum ParamsIOFlag ioFlag);
   virtual void ioParam_phi(enum ParamsIOFlag ioFlag);
   virtual void ioParam_invert(enum ParamsIOFlag ioFlag);

  public:
   InitGaborWeights(char const *name, HyPerCol *hc);
   virtual ~InitGaborWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void calcWeights(int patchIndex, int arborId) override;
   void calcOtherParams(int patchIndex);

  protected:
   InitGaborWeights();
   int initialize(char const *name, HyPerCol *hc);

  private:
   void gaborWeights(float *dataStart);

  private:
   // params variables:
   int mLambda;
   float mPhi;
   bool mInvert;
};

} /* namespace PV */
#endif /* INITGABORWEIGHTS_HPP_ */
