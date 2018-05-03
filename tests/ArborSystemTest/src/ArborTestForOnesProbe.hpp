/*
 * ArborTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestForOnesProbe_HPP_
#define ArborTestForOnesProbe_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class ArborTestForOnesProbe : public PV::StatsProbe {
  public:
   ArborTestForOnesProbe(const char *name, HyPerCol *hc);
   virtual ~ArborTestForOnesProbe();

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* ArborTestForOnesProbe_HPP_ */
