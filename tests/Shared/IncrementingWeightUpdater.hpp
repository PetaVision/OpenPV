/*
 * IncrementingWeightUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef INCREMENTINGWEIGHTUPDATER_HPP_
#define INCREMENTINGWEIGHTUPDATER_HPP_

#include "components/Weights.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

class IncrementingWeightUpdater : public HebbianUpdater {
  public:
   IncrementingWeightUpdater(char const *name, PVParams *params, Communicator const *comm);

   virtual ~IncrementingWeightUpdater() {}

  protected:
   IncrementingWeightUpdater() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual int updateWeights(int arborId) override;
};

} // namespace PV

#endif // INCREMENTINGWEIGHTUPDATER_HPP_
