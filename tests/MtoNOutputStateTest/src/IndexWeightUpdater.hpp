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
   IndexWeightUpdater(char const *name, PVParams *params, Communicator *comm);

   virtual ~IndexWeightUpdater() {}

   virtual void updateState(double timestamp, double dt) override;

  protected:
   IndexWeightUpdater() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   Response::Status initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
};

} // namespace PV

#endif // INDEXWEIGHTUPDATER_HPP_
