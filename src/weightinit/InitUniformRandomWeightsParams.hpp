/*
 * InitUnivormRandomWeightsParams.hpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_
#define INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_

#include "InitRandomWeightsParams.hpp"

namespace PV {

class InitUniformRandomWeightsParams : public PV::InitRandomWeightsParams {
  public:
   InitUniformRandomWeightsParams();
   InitUniformRandomWeightsParams(const char *name, HyPerCol *hc);
   virtual ~InitUniformRandomWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   // get-set methods:
   inline float getWMin() { return mWMin; }
   inline float getWMax() { return mWMax; }
   inline float getSparseFraction() { return mSparseFraction; }
   inline int getMinNNZ() { return mMinNNZ; }

  protected:
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_wMinInit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMaxInit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sparseFraction(enum ParamsIOFlag ioFlag);
   virtual void ioParam_minNNZ(enum ParamsIOFlag ioFlag);

  private:
   float mWMin;
   float mWMax;
   float mSparseFraction; // Percent of zero values in weight patch
   int mMinNNZ; // Minimum number of nonzero values
};

} /* namespace PV */
#endif /* INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_ */
