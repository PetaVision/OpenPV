/*
 * HebbianUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef HEBBIANUPDATER_HPP_
#define HEBBIANUPDATER_HPP_

#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HebbianUpdater : public BaseWeightUpdater {
  protected:
   /**
    * List of parameters needed from the HebbianUpdater class
    * @name HebbianUpdater Parameters
    * @{
    */

   /** @} */ // end of HebbianUpdater parameters

  public:
   HebbianUpdater(char const *name, HyPerCol *hc);

   virtual ~HebbianUpdater() {}

   virtual void updateState(double timestamp, double dt) override;

  protected:
   HebbianUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // HEBBIANUPDATER_HPP_
