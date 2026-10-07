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
   virtual int update_dW(int arborID) override;

   void modifiedUpdateInd_dW(
         int arborID,
         int batchID,
         float const *preLayerData,
         float const *postLayerData,
         long kExt);
}; // end class PlasticTestUpdater

} // end namespace PV
#endif /* PLASTICTESTUPDATER_HPP_ */
