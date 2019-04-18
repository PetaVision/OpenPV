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

class PlasticTestUpdater : public HebbianUpdater {
  public:
   PlasticTestUpdater(const char *name, HyPerCol *hc);
   virtual ~PlasticTestUpdater();

  protected:
   virtual float updateRule_dW(float pre, float post) override;
}; // end class PlasticTestUpdater

} // end namespace PV
#endif /* PLASTICTESTUPDATER_HPP_ */
