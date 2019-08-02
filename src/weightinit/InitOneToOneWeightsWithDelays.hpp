/*
 * InitOneToOneWeightsWithDelays.hpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#ifndef INITONETOONEWEIGHTSWITHDELAYS_HPP_
#define INITONETOONEWEIGHTSWITHDELAYS_HPP_

#include "InitOneToOneWeights.hpp"

namespace PV {

class InitOneToOneWeightsWithDelays : public InitOneToOneWeights {
  public:
   InitOneToOneWeightsWithDelays(char const *name, PVParams *params, Communicator const *comm);
   virtual ~InitOneToOneWeightsWithDelays();

   virtual void calcWeights(int patchIndex, int arborId) override;
   void calcOtherParams(int patchIndex);

  protected:
   InitOneToOneWeightsWithDelays();
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   void
   createOneToOneConnectionWithDelays(float *dataStart, int patchIndex, float iWeight, int arborId);

  protected:
   float mWeightInit = 1.0f;
}; // class InitOneToOneWeightsWightDelays

} // end namespace PV

#endif // INITONETOONEWEIGHTSWITHDELAYS_HPP_
