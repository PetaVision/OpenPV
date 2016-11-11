/*
 * InitCocircWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTS_HPP_
#define INITCOCIRCWEIGHTS_HPP_

#include "InitCocircWeightsParams.hpp"
#include "InitGauss2DWeights.hpp"
#include "InitGauss2DWeightsParams.hpp"
#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitCocircWeightsParams;

class InitCocircWeights : public PV::InitGauss2DWeights {
  public:
   InitCocircWeights(char const *name, HyPerCol *hc);
   virtual ~InitCocircWeights();

   virtual int calcWeights(float *dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams *createNewWeightParams();

  protected:
   InitCocircWeights();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
   bool calcDistChordCocircKurvePreNKurvePost(
         float xDelta,
         float yDelta,
         int kfPost,
         InitCocircWeightsParams *weightParamPtr,
         float thPost);
   int cocircCalcWeights(float *w_tmp, InitCocircWeightsParams *weightParamPtr);
}; // class InitCocircWeights

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTS_HPP_ */
