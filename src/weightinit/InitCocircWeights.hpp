/*
 * InitCocircWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTS_HPP_
#define INITCOCIRCWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"
#include "InitWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"
#include "InitCocircWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitCocircWeightsParams;

class InitCocircWeights: public PV::InitGauss2DWeights {
public:
   InitCocircWeights(char const * name, HyPerCol * hc);
   virtual ~InitCocircWeights();

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();


protected:
   InitCocircWeights();
   int initialize(char const * name, HyPerCol * hc);

private:
   int initialize_base();
   bool calcDistChordCocircKurvePreNKurvePost(
            float xDelta, float yDelta, int kfPost, InitCocircWeightsParams *weightParamPtr, float thPost);
   int cocircCalcWeights(pvdata_t * w_tmp, InitCocircWeightsParams * weightParamPtr);
}; // class InitCocircWeights

BaseObject * createInitCocircWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTS_HPP_ */
