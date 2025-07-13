/*
 * InitOneToOneWeights.hpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#ifndef INITONETOONEWEIGHTS_HPP_
#define INITONETOONEWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

// TODO make InitOneToOneWeights a derived class of InitUniformWeights
class InitOneToOneWeights : public InitWeights {
  protected:
   virtual void ioParam_weightInit(ParamsIOSwitch ioSwitch);

  public:
   InitOneToOneWeights(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~InitOneToOneWeights();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual void calcWeights(int patchIndex, int arborId) override;
   void calcOtherParams(int patchIndex);

  protected:
   InitOneToOneWeights();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   int createOneToOneConnection(float *dataStart, int patchIndex, float weightInit);

  protected:
   float mWeightInit;
}; // class InitOneToOneWeights

} /* namespace PV */
#endif /* INITONETOONEWEIGHTS_HPP_ */
