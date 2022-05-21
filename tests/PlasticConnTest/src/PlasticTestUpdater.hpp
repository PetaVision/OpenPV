/*
 * PlasticTestUpdater.hpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#ifndef PLASTICTESTUPDATER_HPP_
#define PLASTICTESTUPDATER_HPP_

#include <weightupdaters/HebbianUpdater.hpp>

namespace PV {

/**
 * A weight updater used in PlasticConnTest. The update rule pre*post is replaced with pre - post.
 * This allows the plastic conn to be tested over several timesteps without the weights increasing
 * astronomically.
 */
class PlasticTestUpdater : public HebbianUpdater {
  public:
   PlasticTestUpdater(const char *name, PVParams *params, Communicator const *comm);
   virtual ~PlasticTestUpdater();

  protected:
   virtual float updateRule_dW(float pre, float post) override;
}; // end class PlasticTestUpdater

} // end namespace PV
#endif /* PLASTICTESTUPDATER_HPP_ */
