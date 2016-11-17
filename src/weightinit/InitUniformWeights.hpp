/*
 * InitUniformWeights.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTS_HPP_
#define INITUNIFORMWEIGHTS_HPP_

#include "InitUniformWeightsParams.hpp"
#include "InitWeights.hpp"

namespace PV {

class InitUniformWeights : public PV::InitWeights {
  public:
   InitUniformWeights(char const *name, HyPerCol *hc);
   virtual ~InitUniformWeights();
   virtual InitWeightsParams *createNewWeightParams();

   virtual int calcWeights(float *dataStart, int patchIndex, int arborId);

  protected:
   InitUniformWeights();
   int initialize_base();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int uniformWeights(
         float *dataStart,
         float iWeight,
         int kf,
         InitUniformWeightsParams *weightParamPtr,
         bool connectOnlySameFeatures = false);

}; // class InitUniformWeights

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTS_HPP_ */
