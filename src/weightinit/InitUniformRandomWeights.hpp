/*
 * InitUniformRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMRANDOMWEIGHTS_HPP_
#define INITUNIFORMRANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"

namespace PV {

class InitUniformRandomWeights : public PV::InitRandomWeights {
  protected:
   void ioParam_wMinInit(enum ParamsIOFlag ioFlag);
   void ioParam_wMaxInit(enum ParamsIOFlag ioFlag);
   void ioParam_sparseFraction(enum ParamsIOFlag ioFlag);
   void ioParam_minNNZ(enum ParamsIOFlag ioFlag);

  public:
   InitUniformRandomWeights(char const *name, HyPerCol *hc);
   virtual ~InitUniformRandomWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   InitUniformRandomWeights();
   int initialize(char const *name, HyPerCol *hc);
   void randomWeights(float *patchDataStart, int patchIndex) override;

   // Data members
  protected:
   float mWMin           = 0;
   float mWMax           = 1;
   float mSparseFraction = 0; // Percent of zero values in weight patch
   int mMinNNZ           = 0; // Minimum number of nonzero values

}; // class InitUniformRandomWeights

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
