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

class InitUniformRandomWeights : public InitRandomWeights {
  protected:
   void ioParam_wMinInit(ParamsIOSwitch ioSwitch);
   void ioParam_wMaxInit(ParamsIOSwitch ioSwitch);
   void ioParam_sparseFraction(ParamsIOSwitch ioSwitch);
   void ioParam_minNNZ(ParamsIOSwitch ioSwitch);

  public:
   InitUniformRandomWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~InitUniformRandomWeights();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   InitUniformRandomWeights();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual void randomWeights(float *patchDataStart, int patchIndex) override;

   // Data members
  protected:
   float mWMin           = 0;
   float mWMax           = 1;
   float mSparseFraction = 0; // Percent of zero values in weight patch
   int mMinNNZ           = 0; // Minimum number of nonzero values

}; // class InitUniformRandomWeights

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
