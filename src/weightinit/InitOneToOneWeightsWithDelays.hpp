/*
 * InitOneToOneWeightsWithDelays.hpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#ifndef INITONETOONEWEIGHTSWITHDELAYS_HPP_
#define INITONETOONEWEIGHTSWITHDELAYS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitWeightsParams;
class InitOneToOneWeightsWithDelaysParams;

// TODO make InitOneToOneWeightsWithDelays a derived class of InitOneToOneWeights
class InitOneToOneWeightsWithDelays : public PV::InitWeights {
  public:
   InitOneToOneWeightsWithDelays(char const *name, HyPerCol *hc);
   virtual ~InitOneToOneWeightsWithDelays();

   virtual int calcWeights(float *dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams *createNewWeightParams();
   void calcOtherParams(int patchIndex);

  protected:
   InitOneToOneWeightsWithDelays();
   int initialize(char const *name, HyPerCol *hc);
   int createOneToOneConnectionWithDelays(
         float *dataStart,
         int patchIndex,
         float iWeight,
         InitWeightsParams *weightParamPtr,
         int arborId);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* INITONETOONEWEIGHTSWITHDELAYS_HPP_ */
