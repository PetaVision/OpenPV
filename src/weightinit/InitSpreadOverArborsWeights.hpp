/*
 * InitSpreadOverArborsWeights.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTS_HPP_
#define INITSPREADOVERARBORSWEIGHTS_HPP_

#include "InitGauss2DWeights.hpp"
#include "InitWeights.hpp"

namespace PV {

class InitSpreadOverArborsWeightsParams;

class InitSpreadOverArborsWeights : public PV::InitGauss2DWeights {
  public:
   InitSpreadOverArborsWeights(char const *name, HyPerCol *hc);
   virtual ~InitSpreadOverArborsWeights();
   virtual InitWeightsParams *createNewWeightParams();

   virtual int calcWeights(/* PVPatch * patch */ float *dataStart, int patchIndex, int arborId);

  protected:
   InitSpreadOverArborsWeights();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
   int spreadOverArborsWeights(
         /* PVPatch * patch */ float *dataStart,
         int arborId,
         InitSpreadOverArborsWeightsParams *weightParamPtr);
};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTS_HPP_ */
