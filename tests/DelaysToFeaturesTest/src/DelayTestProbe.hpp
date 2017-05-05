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
   DelayTestProbe(const char *probeName, HyPerCol *hc);
   virtual ~DelayTestProbe();

   virtual int outputState(double timestamp);

  protected:
   int initDelayTestProbe(const char *probeName, HyPerCol *hc);

  private:
   int initDelayTestProbe_base();
};

} /* namespace PV */
#endif /* DelayTestProbe_HPP_ */
