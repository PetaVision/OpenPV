/*
 * ArborTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestProbe_HPP_
#define ArborTestProbe_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class ArborTestProbe : public PV::StatsProbe {
  public:
   ArborTestProbe(const char *name, HyPerCol *hc);
   virtual ~ArborTestProbe();

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
