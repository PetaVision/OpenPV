/*
 * LIFGap.hpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#ifndef LIFGAP_HPP_
#define LIFGAP_HPP_

#include "LIF.hpp"

namespace PV {

class LIFGap : public LIF {
  public:
   LIFGap(const char *name, HyPerCol *hc);
   virtual ~LIFGap();

  protected:
   LIFGap();

   int initialize(const char *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // namespace PV

#endif /* LIFGAP_HPP_ */
