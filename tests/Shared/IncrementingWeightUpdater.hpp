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
   IncrementingWeightUpdater(char const *name, HyPerCol *hc);

   virtual ~IncrementingWeightUpdater() {}

  protected:
   IncrementingWeightUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int updateWeights(int arborId);
};

} // namespace PV

#endif // INCREMENTINGWEIGHTUPDATER_HPP_
