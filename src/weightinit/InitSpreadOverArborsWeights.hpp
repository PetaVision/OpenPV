/*
 * InitSpreadOverArborsWeights.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#ifndef INITSPREADOVERARBORSWEIGHTS_HPP_
#define INITSPREADOVERARBORSWEIGHTS_HPP_

#include "InitGauss2DWeights.hpp"

namespace PV {

class InitSpreadOverArborsWeights : public InitGauss2DWeights {
  protected:
   virtual void ioParam_weightInit(ParamsIOSwitch ioSwitch);

  public:
   InitSpreadOverArborsWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~InitSpreadOverArborsWeights();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual void calcWeights(int patchIndex, int arborId) override;

  protected:
   InitSpreadOverArborsWeights();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  private:
   int spreadOverArborsWeights(float *dataStart, int arborId);

  private:
   float mWeightInit = 1.0f;
};

} /* namespace PV */
#endif /* INITSPREADOVERARBORSWEIGHTS_HPP_ */
