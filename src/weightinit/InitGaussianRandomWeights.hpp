/*
 * InitGaussianRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSSIANRANDOMWEIGHTS_HPP_
#define INITGAUSSIANRANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"
#include "columns/GaussianRandom.hpp"

namespace PV {

class InitGaussianRandomWeights : public PV::InitRandomWeights {
  protected:
   void ioParam_wGaussMean(enum ParamsIOFlag ioFlag);
   void ioParam_wGaussStdev(enum ParamsIOFlag ioFlag);

  public:
   InitGaussianRandomWeights(char const *name, HyPerCol *hc);
   virtual ~InitGaussianRandomWeights();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   InitGaussianRandomWeights();
   int initialize(char const *name, HyPerCol *hc);
   virtual int initRNGs(bool isKernel) override;
   virtual void randomWeights(float *patchDataStart, int patchIndex) override;

   // Member variables
  protected:
   GaussianRandom *mGaussianRandState;
   // Use this instead of randState to use Box-Muller transformation.

   float mWGaussMean;
   float mWGaussStdev;
}; // class InitGaussianRandomWeights

} /* namespace PV */

#endif // INITGAUSSIANRANDOMWEIGHTS_HPP_
