/*
 * DelayTestProbe.hpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#ifndef DelayTestProbe_HPP_
#define DelayTestProbe_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class DelayTestProbe : public PV::StatsProbe {
  public:
   DelayTestProbe(const char *name, HyPerCol *hc);
   virtual ~DelayTestProbe();

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* DelayTestProbe_HPP_ */
