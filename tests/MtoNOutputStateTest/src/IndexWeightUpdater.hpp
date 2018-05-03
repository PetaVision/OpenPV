/*
 * IndexWeightUpdater.hpp
 *
 *  Created on: Dec 7, 2017
 *      Author: Pete Schultz
 */

#ifndef INDEXWEIGHTUPDATER_HPP_
#define INDEXWEIGHTUPDATER_HPP_

#include "components/Weights.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

class IndexWeightUpdater : public HebbianUpdater {
  public:
   IndexWeightUpdater(char const *name, HyPerCol *hc);

   virtual ~IndexWeightUpdater() {}

   void initializeWeights();

  protected:
   IndexWeightUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int updateWeights(int arborId) override;
};

} // namespace PV

#endif // INDEXWEIGHTUPDATER_HPP_
