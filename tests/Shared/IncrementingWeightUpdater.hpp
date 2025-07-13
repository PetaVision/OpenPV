/*
 * IncrementingWeightUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef INCREMENTINGWEIGHTUPDATER_HPP_
#define INCREMENTINGWEIGHTUPDATER_HPP_

#include "weightupdaters/HebbianUpdater.hpp"

namespace PV {

class IncrementingWeightUpdater : public HebbianUpdater {
  public:
   IncrementingWeightUpdater(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~IncrementingWeightUpdater() {}

  protected:
   IncrementingWeightUpdater() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual int updateWeights(int arborId) override;
};

} // namespace PV

#endif // INCREMENTINGWEIGHTUPDATER_HPP_
