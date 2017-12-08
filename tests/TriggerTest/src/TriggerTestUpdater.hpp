/*
 * TriggerTestUpdater.hpp
 * Author: peteschultz
 */

#ifndef TRIGGERTESTUPDATER_HPP_
#define TRIGGERTESTUPDATER_HPP_
#include <weightupdaters/HebbianUpdater.hpp>

namespace PV {

class TriggerTestUpdater : public HebbianUpdater {
  public:
   TriggerTestUpdater(const char *name, HyPerCol *hc);

  protected:
   void virtual updateState(double time, double dt) override;
}; // end class TriggerTestUpdater

} // end namespace PV
#endif /* TRIGGERTESTUPDATER_HPP_ */
