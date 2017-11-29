/*
 * InitSpreadOverArborsWeights.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTS_HPP_
#define INITSPREADOVERARBORSWEIGHTS_HPP_

#include "InitGauss2DWeights.hpp"

namespace PV {

class InitSpreadOverArborsWeights : public InitGauss2DWeights {
  protected:
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);

  public:
   InitSpreadOverArborsWeights(char const *name, HyPerCol *hc);
   virtual ~InitSpreadOverArborsWeights();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void calcWeights(int patchIndex, int arborId) override;

  protected:
   InitSpreadOverArborsWeights();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int spreadOverArborsWeights(float *dataStart, int arborId);

  private:
   float mWeightInit = 1.0f;
};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTS_HPP_ */
