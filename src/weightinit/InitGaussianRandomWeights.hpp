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

class InitGaussianRandomWeights : public InitRandomWeights {
  protected:
   void ioParam_wGaussMean(ParamsIOSwitch ioSwitch);
   void ioParam_wGaussStdev(ParamsIOSwitch ioSwitch);

  public:
   InitGaussianRandomWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~InitGaussianRandomWeights();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   InitGaussianRandomWeights();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
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
