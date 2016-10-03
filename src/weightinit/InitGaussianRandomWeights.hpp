/*
 * InitGaussianRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTS_HPP_
#define INITGAUSSIANRANDOMWEIGHTS_HPP_

#include "../columns/GaussianRandom.hpp"
#include "InitGaussianRandomWeightsParams.hpp"
#include "InitRandomWeights.hpp"

namespace PV {

class InitGaussianRandomWeights : public PV::InitRandomWeights {
  public:
   InitGaussianRandomWeights(char const *name, HyPerCol *hc);
   InitGaussianRandomWeights(HyPerConn *conn);
   virtual ~InitGaussianRandomWeights();

   virtual InitWeightsParams *createNewWeightParams();

  protected:
   InitGaussianRandomWeights();
   int initialize(char const *name, HyPerCol *hc);
   virtual int initRNGs(bool isKernel);
   virtual int
   randomWeights(pvdata_t *patchDataStart, InitWeightsParams *weightParamPtr, int patchIndex);

  private:
   int initialize_base();

   // Member variables
  protected:
   GaussianRandom
         *gaussianRandState; // Use this instead of randState to use Box-Muller transformation.
}; // class InitGaussianRandomWeights

} /* namespace PV */
#endif /* INITGAUSSIANRANDOMWEIGHTS_HPP_ */
